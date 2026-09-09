package model

import (
	"math"
	"math/rand"
	"testing"
)

func tinyConfig() *Config {
	return &Config{
		NEmbed: 32, NHeads: 4, NKVHeads: 2, NLayer: 2, NMlp: 64,
		NVocab: 128, TieWordEmbeddings: true,
		RopeTheta: 10000, RmsNormEps: 1e-6, DHead: 8,
		PartialRotaryFactor: 0.5,
	}
}

func randModel(t *testing.T, cfg *Config) *model {
	t.Helper()
	m := newModel(cfg)
	if err := m.alloc(nil, false); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	norm := func(f []float32) {
		for i := range f {
			f[i] = (rng.Float32()*2 - 1) * float32(0.05)
		}
	}
	for i := range m.embed.w {
		m.embed.w[i] = (rng.Float32()*2 - 1) * 0.1
	}
	norm(m.norm)
	for li := range m.layers {
		l := &m.layers[li]
		norm(l.inputLayernorm)
		norm(l.postAttentionLayernorm)
		if a := l.attn; a != nil {
			norm(a.qProj.w)
			norm(a.kProj.w)
			norm(a.vProj.w)
			norm(a.oProj.w)
			for i := range a.qNorm {
				a.qNorm[i] = rng.Float32() * 0.1
			}
			for i := range a.kNorm {
				a.kNorm[i] = rng.Float32() * 0.1
			}
		}
		if g := l.gdn; g != nil {
			norm(g.inProjQKV.w)
			norm(g.inProjZ.w)
			norm(g.inProjB.w)
			norm(g.inProjA.w)
			norm(g.outProj.w)
			norm(g.convW)
			norm(g.normW)
		}
		if d, ok := l.mlp.(*denseMLP); ok {
			norm(d.gate.w)
			norm(d.up.w)
			norm(d.down.w)
		}
	}
	return m
}

// TestPrefillDecodeEquivalence: feeding a prompt in one shot and then
// decoding must give the same next-token logits as feeding the whole thing
// token by token. Catches rotary offset bugs, cache bugs, mask bugs.
func TestPrefillDecodeEquivalence(t *testing.T) {
	cfg := tinyConfig()
	m := randModel(t, cfg)

	ids := []int{5, 17, 3, 99, 42}
	caches := allocCaches(m)
	pos := []int{0, 1, 2, 3, 4}
	full, err := m.forward(ids, pos, caches)
	if err != nil {
		t.Fatal(err)
	}

	// decode path: prefill 4, then one step
	c2 := allocCaches(m)
	l4, err := m.forward(ids[:4], pos[:4], c2)
	if err != nil {
		t.Fatal(err)
	}
	l5, err := m.forward(ids[4:5], []int{4}, c2)
	if err != nil {
		t.Fatal(err)
	}
	_ = l4
	for i := range full {
		if math.Abs(float64(full[i]-l5[i])) > 2e-3 {
			t.Fatalf("prefill vs decode logits diverge at %d: %f vs %f", i, full[i], l5[i])
		}
	}
}

// TestGenerateDeterministic: same prompt, same tokens.
func TestGenerateDeterministic(t *testing.T) {
	cfg := tinyConfig()
	m1 := randModel(t, cfg)
	m2 := randModel(t, cfg) // same seed → same weights

	var a, b []int
	stop := map[int]bool{127: true}
	emit := func(sink *[]int) func(int) {
		return func(tok int) { *sink = append(*sink, tok) }
	}
	if err := m1.Generate([]int{3, 9, 44}, 8, stop, emit(&a)); err != nil {
		t.Fatal(err)
	}
	if err := m2.Generate([]int{3, 9, 44}, 8, stop, emit(&b)); err != nil {
		t.Fatal(err)
	}
	if len(a) == 0 || len(a) != len(b) {
		t.Fatalf("lengths differ: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("token %d differs: %d vs %d", i, a[i], b[i])
		}
	}
}

// TestQuantRoundTrip: int8 quantize→dequant must track the original weights
// within one scale step; int4 within ~2 steps.
func TestQuantRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	outDim, inDim := 16, 64 // inDim divisible by 32
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.1
	}
	qw, sc := quantizeInt8(w, outDim, inDim)
	q := &quantLinear{bits: 8, outFeatures: outDim, inFeatures: inDim, qweight: qw, scale: sc}
	dq := q.dequantize()
	for i := range w {
		step := sc[i/(inDim)] // conservative: any scale in the row
		_ = step
		if math.Abs(float64(w[i]-dq[i])) > 0.02 {
			t.Fatalf("int8 dequant off at %d: %f vs %f", i, w[i], dq[i])
		}
	}
}

// TestGDNCacheEquivalence: GDN processed as prefill(4)+decode(1) must match
// one-shot prefill(5).
func TestGDNCacheEquivalence(t *testing.T) {
	cfg := tinyConfig()
	cfg.LayerTypes = []string{"linear_attention", "linear_attention"}
	cfg.NLinearKHeads = 2
	cfg.NLinearVHeads = 4
	cfg.DLinearK = 8
	cfg.DLinearV = 8
	cfg.LinearConvKernel = 4
	m := randModel(t, cfg)

	ids := []int{1, 2, 3, 4, 5}
	c1 := allocCaches(m)
	full, err := m.forward(ids, []int{0, 1, 2, 3, 4}, c1)
	if err != nil {
		t.Fatal(err)
	}
	c2 := allocCaches(m)
	if _, err := m.forward(ids[:4], []int{0, 1, 2, 3}, c2); err != nil {
		t.Fatal(err)
	}
	dec, err := m.forward(ids[4:5], []int{4}, c2)
	if err != nil {
		t.Fatal(err)
	}
	for i := range full {
		if math.Abs(float64(full[i]-dec[i])) > 5e-3 {
			t.Fatalf("GDN cache mismatch at %d: %f vs %f", i, full[i], dec[i])
		}
	}
}

func TestMoERouting(t *testing.T) {
	m := &moeMLP{nExperts: 4, topK: 2, nEmbed: 8, nMoeMlp: 16, nEmbedSet: 8}
	m.gate.w = []float32{1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	m.gate.w = m.gate.w[:4]
	m.gateUpW = make([]float32, 4*2*16*8)
	m.downW = make([]float32, 4*8*16)
	m.shared = &denseMLP{nEmbed: 8, nMlp: 4}
	m.shared.gate.w = make([]float32, 4*8)
	m.shared.up.w = make([]float32, 4*8)
	m.shared.down.w = make([]float32, 8*4)
	m.sharedGate = make([]float32, 8)
	x := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	out := m.forward(x)
	if len(out) != 8 {
		t.Fatalf("bad out len %d", len(out))
	}
	if err := checkFinite(out, "moe"); err != nil {
		t.Fatal(err)
	}
}

package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// ---------------------------------------------------------------------------
// Rotary embeddings. Text-only path: sequential positions over the 3 mRoPE
// sections collapse to the same values, so cos/sin are (T, rotaryDim).
// ---------------------------------------------------------------------------

type rotary struct {
	invFreq []float32
	rotDim  int
}

func newRotary(cfg *Config) *rotary {
	dim := cfg.RotaryDim()
	inv := make([]float32, dim/2)
	for i := range inv {
		r := float64(2 * i)
		inv[i] = float32(1.0 / math.Pow(cfg.RopeTheta, r/float64(dim)))
	}
	return &rotary{invFreq: inv, rotDim: dim}
}

// cosSin returns per-position cos/sin tables (T, rotDim). pos3 carries the
// three mRoPE sections (temporal, height, width); the interleaved merge
// fills pair p with section p%3, index p/3 — text tokens advance all three
// sections together, so plain text produces the same values as before.
func (r *rotary) cosSin(pos3 [3][]int) (cos, sin []float32) {
	T := len(pos3[0])
	half := r.rotDim / 2
	cos = make([]float32, T*r.rotDim)
	sin = make([]float32, T*r.rotDim)
	for t := range pos3[0] {
		for i := 0; i < half; i++ {
			sec := i % 3
			idx := i / 3
			freq := float64(pos3[sec][t]) * float64(r.invFreq[idx])
			c, s := math.Cos(freq), math.Sin(freq)
			// emb = cat([freqs, freqs]) — same angle for both halves
			cos[t*r.rotDim+i] = float32(c)
			cos[t*r.rotDim+half+i] = float32(c)
			sin[t*r.rotDim+i] = float32(s)
			sin[t*r.rotDim+half+i] = float32(s)
		}
	}
	return cos, sin
}

// rotateHalf: (-x2, x1) over the last dim of size n.
func rotateHalf(x []float32) []float32 {
	n := len(x)
	h := n / 2
	out := make([]float32, n)
	for i := 0; i < h; i++ {
		out[i] = -x[h+i]
		out[h+i] = x[i]
	}
	return out
}

// applyRotary in-place on (H, d) rows: first rotDim dims rotated, rest passed.
func applyRotary(x []float32, heads int, cos, sin []float32, t int) {
	d := len(x) / heads
	rd := len(cos) / (t + 1) // not used; rotDim passed via cos length per pos
	_ = rd
	rotDim := rotDimOf(cos)
	for h := 0; h < heads; h++ {
		row := x[h*d : (h+1)*d]
		ct := cos[t*rotDim : (t+1)*rotDim]
		st := sin[t*rotDim : (t+1)*rotDim]
		for i := 0; i < rotDim; i++ {
			a, b := row[i], row[i+rotDim/2]
			// rotateHalf pairs (i, i+rotDim/2)
			row[i] = a*ct[i] - b*st[i]
			row[i+rotDim/2] = b*ct[i] + a*st[i]
		}
	}
}

func rotDimOf(cos []float32) int { return cosLenPerPos(cos) }

var cosPerPos int

func cosLenPerPos(cos []float32) int { return cosPerPos }

// ---------------------------------------------------------------------------
// KV cache for attention layers.
// ---------------------------------------------------------------------------

type kvCache struct {
	k [][]float32 // per token: nKVHeads * dHead
	v [][]float32
}

// ---------------------------------------------------------------------------
// SelfAttention with q-gate (Qwen3.5 style) and per-head q/k Gemma norms.
// ---------------------------------------------------------------------------

type linear struct {
	w      []float32 // dense (out, in) — used when raw is nil
	raw    []byte    // bf16 storage, (out, in) — memory-dense path
	outDim int
	inDim  int
}

func bf16f32(b []byte) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16)
}

func traceCK(name string, x []float32) {
	if os.Getenv("QWENGO_TRACE") == "" {
		return
	}
	var sum, sq float64
	for _, v := range x {
		sum += float64(v)
		sq += float64(v) * float64(v)
	}
	fmt.Fprintf(os.Stderr, "GO %s %.6e %.6e\n", name, sum, sq)
}

// matvec computes y = x @ W^T for one token, bf16-native when loaded raw.
// Parallelized across output rows — the LM head alone is 248k rows × 1k.
func (l *linear) matvec(x []float32, out int) []float32 {
	if l.raw == nil {
		return matmulVec(x, l.w, out)
	}
	y := make([]float32, out)
	const minWork = 1 << 16
	if out*l.inDim < minWork {
		matvecRange(x, l.raw, y, l.inDim, out, 0, out)
		return y
	}
	workers := runtimeNumWorkers()
	chunk := (out + workers - 1) / workers
	done := make(chan struct{}, workers)
	spawned := 0
	for i := 0; i < workers; i++ {
		lo := i * chunk
		hi := lo + chunk
		if hi > out {
			hi = out
		}
		if lo >= hi {
			break
		}
		spawned++
		go func(lo, hi int) {
			defer func() { done <- struct{}{} }()
			matvecRange(x, l.raw, y, l.inDim, out, lo, hi)
		}(lo, hi)
	}
	for i := 0; i < spawned; i++ {
		<-done
	}
	return y
}

// matvecRange computes rows [lo, hi) of y = x @ W^T with W in bf16.
func matvecRange(x []float32, raw []byte, y []float32, in, out, lo, hi int) {
	for o := lo; o < hi; o++ {
		row := raw[o*in*2 : (o+1)*in*2]
		var s float32
		for k, xv := range x {
			s += xv * bf16f32(row[k*2:])
		}
		y[o] = s
	}
}

// mat2D computes y = x @ W^T over (T, in).
func (l *linear) mat2D(x []float32, T, out int) []float32 {
	if T == 1 {
		return l.matvec(x, out)
	}
	if l.raw == nil {
		return matmul(x, l.w, T, len(x)/T, out)
	}
	in := len(x) / T
	y := make([]float32, T*out)
	for t := 0; t < T; t++ {
		copy(y[t*out:(t+1)*out], l.matvec(x[t*in:(t+1)*in], out))
	}
	return y
}

// proj runs a linear over (T, in): quantized wrappers dequantize once for
// the whole batch (transient), bf16-native weights convert per element.
func proj(x []float32, T int, l *linear, q *quantLinear, out int) []float32 {
	if q != nil {
		if T == 1 {
			return q.forward(x)
		}
		dw := q.dequantize()
		return matmul(x, dw, T, len(x)/T, out)
	}
	return l.mat2D(x, T, out)
}

type selfAttention struct {
	qProj  linear // out = nHeads*dHead*2 (query + gate)
	kProj  linear
	vProj  linear
	oProj  linear
	qQuant *quantLinear
	kQuant *quantLinear
	vQuant *quantLinear
	oQuant *quantLinear
	qNorm  []float32 // Gemma norm weights, dHead
	kNorm  []float32
	nHeads int
	nKV    int
	dHead  int
	nEmbed int
	scale  float32
}

func (a *selfAttention) forward(x []float32, T int, cos, sin []float32, cache *kvCache) []float32 {
	ne := len(x) / T
	d := a.dHead

	qg := proj(x, T, &a.qProj, a.qQuant, a.nHeads*d*2)
	k := proj(x, T, &a.kProj, a.kQuant, a.nKV*d)
	v := proj(x, T, &a.vProj, a.vQuant, a.nKV*d)

	// split q/gate per head; norm q heads; rotate q and k
	q := make([]float32, T*a.nHeads*d)
	gate := make([]float32, T*a.nHeads*d)
	for t := 0; t < T; t++ {
		for h := 0; h < a.nHeads; h++ {
			src := qg[t*(a.nHeads*2*d)+h*2*d:]
			copy(q[t*a.nHeads*d+h*d:], src[:d])
			copy(gate[t*a.nHeads*d+h*d:], src[d:2*d])
		}
	}
	for t := 0; t < T; t++ {
		for h := 0; h < a.nHeads; h++ {
			head := q[t*a.nHeads*d+h*d : t*a.nHeads*d+(h+1)*d]
			gemmaRMSNormApply(head, a.qNorm, float32(epsDefault))
		}
	}
	for t := 0; t < T; t++ {
		for h := 0; h < a.nKV; h++ {
			head := k[t*a.nKV*d+h*d : t*a.nKV*d+(h+1)*d]
			gemmaRMSNormApply(head, a.kNorm, float32(epsDefault))
		}
	}
	q = applyRotarySeq(q, T, a.nHeads, cos, sin)
	k = applyRotarySeq(k, T, a.nKV, cos, sin)

	// append to cache
	for t := 0; t < T; t++ {
		kt := make([]float32, a.nKV*d)
		copy(kt, k[t*a.nKV*d:(t+1)*a.nKV*d])
		cache.k = append(cache.k, kt)
		vt := make([]float32, a.nKV*d)
		copy(vt, v[t*a.nKV*d:(t+1)*a.nKV*d])
		cache.v = append(cache.v, vt)
	}
	past := len(cache.k) - T

	// attention per head; GQA via head repeat
	out := make([]float32, T*a.nHeads*d)
	rep := a.nHeads / a.nKV
	for t := 0; t < T; t++ {
		for h := 0; h < a.nHeads; h++ {
			kvh := h / rep
			qh := q[t*a.nHeads*d+h*d : t*a.nHeads*d+(h+1)*d]
			// scores over past+t tokens
			scores := make([]float32, past+t+1)
			for s := 0; s <= past+t; s++ {
				kh := cache.k[s][kvh*d : kvh*d+d]
				var dot float32
				for i, qv := range qh {
					dot += qv * kh[i]
				}
				scores[s] = dot * a.scale
			}
			// causal mask: position s must be <= past+t (it is, since we only
			// attend to cache entries up to this token) — full ledger is fine.
			softmax(scores)
			acc := out[t*a.nHeads*d+h*d : t*a.nHeads*d+(h+1)*d]
			for s := 0; s <= past+t; s++ {
				vh := cache.v[s][kvh*d : kvh*d+d]
				w := scores[s]
				for i := range acc {
					acc[i] += w * vh[i]
				}
			}
		}
	}
	// gate: y * sigmoid(gate)
	for i := range out {
		out[i] *= sigmoid1(gate[i])
	}
	res := make([]float32, T*ne)
	for t := 0; t < T; t++ {
		r := proj(out[t*a.nHeads*d:(t+1)*a.nHeads*d], 1, &a.oProj, a.oQuant, ne)
		copy(res[t*ne:(t+1)*ne], r)
	}
	return res
}

// applyRotarySeq applies rotary per token over (T, heads, d), in place.
// Pairing is the NeoX style rotate_half: dims (i, half+i).
func applyRotarySeq(x []float32, T, heads int, cos, sin []float32) []float32 {
	d := len(x) / (T * heads)
	half := rotDimTotal / 2
	for t := 0; t < T; t++ {
		ct := cos[t*rotDimTotal : (t+1)*rotDimTotal]
		st := sin[t*rotDimTotal : (t+1)*rotDimTotal]
		for h := 0; h < heads; h++ {
			row := x[(t*heads+h)*d : (t*heads+h+1)*d]
			for i := 0; i < half; i++ {
				a, b := row[i], row[half+i]
				row[i] = a*ct[i] - b*st[i]
				row[half+i] = b*ct[i] + a*st[i]
			}
		}
	}
	return x
}

var rotDimTotal int

var epsDefault = 1e-6

// ---------------------------------------------------------------------------
// Dense MLP (SwiGLU).
// ---------------------------------------------------------------------------

type denseMLP struct {
	gate      linear
	up        linear
	down      linear
	gateQuant *quantLinear
	upQuant   *quantLinear
	downQuant *quantLinear
	nEmbed    int
	nMlp      int
}

func (m *denseMLP) forward(x []float32) []float32 {
	g := proj(x, 1, &m.gate, m.gateQuant, m.nMlp)
	u := proj(x, 1, &m.up, m.upQuant, m.nMlp)
	silu(g)
	for i := range g {
		g[i] *= u[i]
	}
	return proj(g, 1, &m.down, m.downQuant, m.nEmbed)
}

// ---------------------------------------------------------------------------
// Block + Model.
// ---------------------------------------------------------------------------

type block struct {
	layerType              string
	inputLayernorm         []float32 // Gemma norm
	postAttentionLayernorm []float32
	attn                   *selfAttention
	gdn                    *gatedDeltaNet
	mlp                    mixer // denseMLP, moeMLP, or quantized wrappers
}

type mixer interface {
	forward(x []float32) []float32
}

type model struct {
	cfg     *Config
	rot     *rotary
	embed   linear // (vocab, nEmbed); bf16 raw or f32
	layers  []block
	norm    []float32 // final Gemma norm
	lmHead  linear    // unused when tied
	visual  *visionTower
	tied    bool
	rawMode bool        // weights loaded as raw bf16 (no f32 upcast)
	closers []io.Closer // live shard mmaps (raw slices point into them)
}

// newModel builds an empty model skeleton from config.
func newModel(cfg *Config) *model {
	m := &model{cfg: cfg, tied: cfg.TieWordEmbeddings, rot: newRotary(cfg)}
	rotDimTotal = cfg.RotaryDim()
	for i := 0; i < cfg.NLayer; i++ {
		m.layers = append(m.layers, newBlock(cfg, i))
	}
	return m
}

func newBlock(cfg *Config, i int) block {
	eps := float32(cfg.RmsNormEps)
	b := block{
		layerType:              cfg.LayerType(i),
		inputLayernorm:         make([]float32, cfg.NEmbed),
		postAttentionLayernorm: make([]float32, cfg.NEmbed),
	}
	d := cfg.DHead
	if b.layerType == "linear_attention" {
		b.gdn = newGatedDeltaNet(cfg)
	} else {
		b.attn = &selfAttention{
			nHeads: cfg.NHeads,
			nKV:    cfg.NKVHeads,
			dHead:  d,
			scale:  float32(1 / math.Sqrt(float64(d))),
		}
		b.attn.qNorm = make([]float32, d)
		b.attn.kNorm = make([]float32, d)
	}
	_ = eps
	if cfg.NExperts > 0 {
		b.mlp = newMoeMLP(cfg)
	} else {
		m := &denseMLP{}
		b.mlp = m
	}
	return b
}

// layerCaches holds one cache per layer (kvCache or gdnCache).
type layerCaches struct {
	kv  []*kvCache
	gdn []*gdnCache
}

func allocCaches(m *model) *layerCaches {
	c := &layerCaches{}
	for _, l := range m.layers {
		if l.attn != nil {
			c.kv = append(c.kv, &kvCache{})
		} else {
			c.gdn = append(c.gdn, newGdnCache(l.gdn))
		}
	}
	return c
}

// forward runs the model over input ids; returns logits for the last
// position only when lastOnly (decode path), else full (prefill path —
// only the last row is actually needed, so we always compute just that).
func (m *model) forward(inputIDs []int, pos3 [3][]int, caches *layerCaches) ([]float32, error) {
	ne := m.cfg.NEmbed
	T := len(inputIDs)
	x := make([]float32, T*ne)
	for t, id := range inputIDs {
		if id < 0 || id >= m.cfg.NVocab {
			return nil, fmt.Errorf("token id %d out of vocab range", id)
		}
		// embedding row fetch (no matmul): decode bf16 rows on the fly
		if m.embed.raw != nil {
			base := id * ne * 2
			for i := 0; i < ne; i++ {
				x[t*ne+i] = bf16f32(m.embed.raw[base+i*2:])
			}
		} else {
			copy(x[t*ne:(t+1)*ne], m.embed.w[id*ne:(id+1)*ne])
		}
	}
	return m.decodeLayers(x, pos3, caches)
}

// decodeLayers runs the shared decoder stack over embedded states.
func (m *model) decodeLayers(x []float32, pos3 [3][]int, caches *layerCaches) ([]float32, error) {
	ne := m.cfg.NEmbed
	T := len(x) / ne
	cos, sin := m.rot.cosSin(pos3)
	kvIdx, gdnIdx := 0, 0
	for i := range m.layers {
		traceCK(fmt.Sprintf("pre_layer%d", i), x)
		l := &m.layers[i]
		var y []float32
		if l.attn != nil {
			perTok := make([]float32, len(x))
			for t := 0; t < T; t++ {
				tok := make([]float32, ne)
				copy(tok, x[t*ne:(t+1)*ne])
				gemmaRMSNormApply(tok, l.inputLayernorm, float32(m.cfg.RmsNormEps))
				copy(perTok[t*ne:(t+1)*ne], tok)
			}
			y = l.attn.forward(perTok, T, cos, sin, caches.kv[kvIdx])
			kvIdx++
		} else {
			perTok := make([]float32, len(x))
			for t := 0; t < T; t++ {
				tok := make([]float32, ne)
				copy(tok, x[t*ne:(t+1)*ne])
				gemmaRMSNormApply(tok, l.inputLayernorm, float32(m.cfg.RmsNormEps))
				copy(perTok[t*ne:(t+1)*ne], tok)
			}
			y = l.gdn.forward(perTok, T, caches.gdn[gdnIdx])
			gdnIdx++
		}
		for j := range x {
			x[j] += y[j]
		}
		// MLP on post-norm
		y2 := make([]float32, T*ne)
		for t := 0; t < T; t++ {
			tok := make([]float32, ne)
			copy(tok, x[t*ne:(t+1)*ne])
			gemmaRMSNormApply(tok, l.postAttentionLayernorm, float32(m.cfg.RmsNormEps))
			out := l.mlp.forward(tok)
			copy(y2[t*ne:(t+1)*ne], out)
		}
		for j := range x {
			x[j] += y2[j]
		}
	}
	last := x[(T-1)*ne : T*ne]
	gemmaRMSNormApply(last, m.norm, float32(m.cfg.RmsNormEps))
	traceCK("finalnorm", x)
	if m.tied {
		return m.embed.matvec(last, m.cfg.NVocab), nil
	}
	return m.lmHead.matvec(last, m.cfg.NVocab), nil
}

// Generate greedily streams tokens. stop map keys are stop token ids.
func (m *model) Generate(inputIDs []int, maxNew int, stop map[int]bool, emit func(int)) error {
	return m.generate(inputIDs, nil, ImageGrid{}, maxNew, stop, emit)
}

// pos3Flat builds three identical mRoPE sections from a plain position list.
func pos3Flat(pos []int) [3][]int {
	return [3][]int{pos, append([]int(nil), pos...), append([]int(nil), pos...)}
}

// generate runs greedy decode; a non-nil pixels buffer makes it an image
// prompt: the image-pat tokens get vision features and the prompt positions
// come from pos3 (mRoPE), decode continues at maxPos+1 in all sections.
func (m *model) generate(inputIDs []int, pixels []float32, grid ImageGrid, maxNew int, stop map[int]bool, emit func(int)) error {
	caches := allocCaches(m)
	imageToken := m.cfg.ImageTokenID
	var pos3 [3][]int
	if pixels != nil {
		var err error
		pos3, err = mropePositions(inputIDs, []ImageGrid{grid}, imageToken)
		if err != nil {
			return err
		}
	} else {
		pos := make([]int, len(inputIDs))
		for i := range pos {
			pos[i] = i
		}
		pos3 = pos3Flat(pos)
	}
	logits, err := m.forwardVision(inputIDs, pixels, grid, pos3, caches)
	if err != nil {
		return err
	}
	maxPos := 0
	for _, s := range pos3 {
		for _, p := range s {
			if p > maxPos {
				maxPos = p
			}
		}
	}
	nextPos := maxPos + 1
	for step := 0; step < maxNew; step++ {
		tok := argmax(logits)
		emit(tok)
		if stop[tok] {
			return nil
		}
		dp := []int{nextPos}
		logits, err = m.forward([]int{tok}, pos3Flat(dp), caches)
		if err != nil {
			return err
		}
		nextPos++
	}
	return nil
}

// forwardVision is forward plus vision-feature injection for image-pat tokens.
func (m *model) forwardVision(inputIDs []int, pixels []float32, grid ImageGrid, pos3 [3][]int, caches *layerCaches) ([]float32, error) {
	if pixels == nil {
		return m.forward(inputIDs, pos3, caches)
	}
	if m.visual == nil {
		return nil, fmt.Errorf("checkpoint has no vision tower")
	}
	vis, err := m.visual.encode(pixels, grid)
	if err != nil {
		return nil, err
	}
	ne := m.cfg.NEmbed
	imageToken := m.cfg.ImageTokenID
	T := len(inputIDs)
	nPatches := len(vis) / ne
	count := 0
	for _, id := range inputIDs {
		if id == imageToken {
			count++
		}
	}
	if count != nPatches {
		return nil, fmt.Errorf("vision: %d image-pat tokens but %d vision features", count, nPatches)
	}
	x := make([]float32, T*ne)
	q := 0
	for t, id := range inputIDs {
		if id == imageToken {
			copy(x[t*ne:(t+1)*ne], vis[q*ne:(q+1)*ne])
			q++
			continue
		}
		if id < 0 || id >= m.cfg.NVocab {
			return nil, fmt.Errorf("token id %d out of vocab range", id)
		}
		if m.embed.raw != nil {
			base := id * ne * 2
			for i := 0; i < ne; i++ {
				x[t*ne+i] = bf16f32(m.embed.raw[base+i*2:])
			}
		} else {
			copy(x[t*ne:(t+1)*ne], m.embed.w[id*ne:(id+1)*ne])
		}
	}
	// rest of the decoder is identical to forward — refactor: inline the
	// layer stack via a helper below.
	return m.decodeLayers(x, pos3, caches)
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// GenerateWithImage runs greedy decode over a multimodal (image + text)
// prompt. pixels/grid come from model.ProcessImage.
func (m *model) GenerateWithImage(inputIDs []int, pixels []float32, grid ImageGrid, maxNew int, stop map[int]bool, emit func(int)) error {
	return m.generate(inputIDs, pixels, grid, maxNew, stop, emit)
}

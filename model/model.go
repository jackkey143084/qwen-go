package model

import (
	"fmt"
	"math"
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

// cosSin returns per-position cos/sin tables (T, rotDim).
func (r *rotary) cosSin(positions []int) (cos, sin []float32) {
	T := len(positions)
	half := r.rotDim / 2
	cos = make([]float32, T*r.rotDim)
	sin = make([]float32, T*r.rotDim)
	for t, p := range positions {
		for i := 0; i < half; i++ {
			freq := float64(p) * float64(r.invFreq[i])
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
	w []float32 // (out, in)
}

// proj runs a linear over (T, in): quantized wrappers dequantize once for
// the whole batch (transient), dense weights use the batched matmul.
func proj(x []float32, T int, w []float32, q *quantLinear, out int) []float32 {
	in := len(x) / T
	if q != nil {
		if T == 1 {
			return q.forward(x)
		}
		dw := q.dequantize()
		return matmul(x, dw, T, in, out)
	}
	if T == 1 {
		return matmulVec(x, w, out)
	}
	return matmul(x, w, T, in, out)
}

type selfAttention struct {
	qProj   linear // out = nHeads*dHead*2 (query + gate)
	kProj   linear
	vProj   linear
	oProj   linear
	qQuant  *quantLinear
	kQuant  *quantLinear
	vQuant  *quantLinear
	oQuant  *quantLinear
	qNorm   []float32 // Gemma norm weights, dHead
	kNorm   []float32
	nHeads  int
	nKV     int
	dHead   int
	nEmbed  int
	scale   float32
}

func (a *selfAttention) forward(x []float32, T int, cos, sin []float32, cache *kvCache) []float32 {
	ne := len(x) / T
	d := a.dHead

	qg := proj(x, T, a.qProj.w, a.qQuant, a.nHeads*d*2)
	k := proj(x, T, a.kProj.w, a.kQuant, a.nKV*d)
	v := proj(x, T, a.vProj.w, a.vQuant, a.nKV*d)

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
		r := proj(out[t*a.nHeads*d:(t+1)*a.nHeads*d], 1, a.oProj.w, a.oQuant, ne)
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
	g := proj(x, 1, m.gate.w, m.gateQuant, m.nMlp)
	u := proj(x, 1, m.up.w, m.upQuant, m.nMlp)
	silu(g)
	for i := range g {
		g[i] *= u[i]
	}
	return proj(g, 1, m.down.w, m.downQuant, m.nEmbed)
}

// ---------------------------------------------------------------------------
// Block + Model.
// ---------------------------------------------------------------------------

type block struct {
	layerType             string
	inputLayernorm        []float32 // Gemma norm
	postAttentionLayernorm []float32
	attn                  *selfAttention
	gdn                   *gatedDeltaNet
	mlp                   mixer // denseMLP, moeMLP, or quantized wrappers
}

type mixer interface {
	forward(x []float32) []float32
}

type model struct {
	cfg    *Config
	rot    *rotary
	embed  []float32 // (vocab, nEmbed)
	layers []block
	norm   []float32 // final Gemma norm
	lmHead []float32 // nil when tied
	tied   bool
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
func (m *model) forward(inputIDs []int, pos []int, caches *layerCaches) ([]float32, error) {
	ne := m.cfg.NEmbed
	T := len(inputIDs)
	x := make([]float32, T*ne)
	for t, id := range inputIDs {
		if id < 0 || id >= m.cfg.NVocab {
			return nil, fmt.Errorf("token id %d out of vocab range", id)
		}
		copy(x[t*ne:(t+1)*ne], m.embed[id*ne:(id+1)*ne])
	}
	cos, sin := m.rot.cosSin(pos)
	kvIdx, gdnIdx := 0, 0
	for i := range m.layers {
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
	if m.tied {
		return matmulVec(last, m.embed, m.cfg.NVocab), nil
	}
	return matmulVec(last, m.lmHead, m.cfg.NVocab), nil
}

// Generate greedily streams tokens. stop map keys are stop token ids.
func (m *model) Generate(inputIDs []int, maxNew int, stop map[int]bool, emit func(int)) error {
	caches := allocCaches(m)
	pos := make([]int, len(inputIDs))
	for i := range pos {
		pos[i] = i
	}
	logits, err := m.forward(inputIDs, pos, caches)
	if err != nil {
		return err
	}
	nextPos := len(inputIDs)
	for step := 0; step < maxNew; step++ {
		tok := argmax(logits)
		emit(tok)
		if stop[tok] {
			return nil
		}
		logits, err = m.forward([]int{tok}, []int{nextPos}, caches)
		if err != nil {
			return err
		}
		nextPos++
	}
	return nil
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

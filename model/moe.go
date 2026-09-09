package model

import "math"

// ---------------------------------------------------------------------------
// MoE MLP — router (softmax f32), top-k, renormalized weights, batched
// stacked experts, optional gated shared expert. Port of MoEMLP/MoEExperts.
// ---------------------------------------------------------------------------

type moeMLP struct {
	gate            linear // (nExperts, nEmbed)
	gateUpW         []float32 // (nExperts, 2*nMoeMlp, nEmbed)
	downW           []float32 // (nExperts, nEmbed, nMoeMlp)
	routerQuant     *quantLinear
	sharedGateQuant *quantLinear
	sharedGate      []float32 // (1, nEmbed) or nil
	shared          *denseMLP // or nil
	nExperts        int
	topK            int
	nEmbed          int
	nEmbedSet       int
	nMoeMlp         int
}

func newMoeMLP(cfg *Config) *moeMLP {
	m := &moeMLP{
		nExperts: cfg.NExperts,
		topK:     cfg.NExpertsPerToken,
		nEmbed:   cfg.NEmbed,
		nMoeMlp:  cfg.NMoeMlp,
	}
	if cfg.NSharedExpertMlp > 0 {
		m.shared = &denseMLP{nEmbed: cfg.NEmbed, nMlp: cfg.NSharedExpertMlp}
		m.sharedGate = make([]float32, cfg.NEmbed)
	}
	return m
}

func (m *moeMLP) forward(x []float32) []float32 {
	logits := proj(x, 1, m.gate.w, m.routerQuant, m.nExperts)
	softmax(logits)
	// top-k by selection (k is small)
	idx := make([]int, m.nExperts)
	for i := range idx {
		idx[i] = i
	}
	for i := 1; i < m.topK && i < len(idx); i++ {
		for j := i; j > 0 && logits[idx[j]] > logits[idx[j-1]]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}
	wSum := float32(0)
	for i := 0; i < m.topK; i++ {
		wSum += logits[idx[i]]
	}
	out := make([]float32, m.nEmbed)
	guStride := 2 * m.nMoeMlp * m.nEmbed
	dStr := m.nEmbed * m.nMoeMlp
	for i := 0; i < m.topK; i++ {
		e := idx[i]
		w := logits[e] / (wSum + 1e-9)
		gu := matmulVec(x, m.gateUpW[e*guStride:(e+1)*guStride], m.nMoeMlp*2)
		g := gu[:m.nMoeMlp]
		u := gu[m.nMoeMlp:]
		silu(g)
		for j := range g {
			g[j] *= u[j]
		}
		d := matmulVec(g, m.downW[e*dStr:(e+1)*dStr], m.nEmbed)
		for j := range out {
			out[j] += w * d[j]
		}
	}
	if m.shared != nil {
		s := m.shared.forward(x)
		// sigmoid(shared_gate · x) — gate weight is (1, nEmbed)
		var g float32
		for i := range x {
			g += x[i] * m.sharedGate[i]
		}
		sg := sigmoid1(g)
		for j := range out {
			out[j] += sg * s[j]
		}
	}
	return out
}

// set by the loader; strides per expert
var _ = 0

// ---------------------------------------------------------------------------
// Weight-only block quantization, ported from tiny-qwen: 32 weights share
// one fp16 scale; int8 or int4 (packed two per byte). Buys model FIT, not
// speed — dequant is transient per forward.
// ---------------------------------------------------------------------------

const quantGroup = 32

type quantLinear struct {
	bits        int
	outFeatures int
	inFeatures  int
	qweight     []byte // int8 codes (bits=8) or packed pairs (bits=4)
	scale       []float32 // f16 loaded as f32, (out, in/32)
}

func (q *quantLinear) dequantize() []float32 {
	outDim, inDim := q.outFeatures, q.inFeatures
	w := make([]float32, outDim*inDim)
	if q.bits == 8 {
		for o := 0; o < outDim; o++ {
			for c := 0; c < inDim; c++ {
				var v int8 = int8(q.qweight[o*inDim+c])
				w[o*inDim+c] = float32(v) * q.scale[o*(inDim/quantGroup)+c/quantGroup]
			}
		}
		return w
	}
	// int4: hi nibble first, low nibble second, stored +8
	for o := 0; o < outDim; o++ {
		for c := 0; c < inDim/2; c++ {
			b := q.qweight[o*(inDim/2)+c]
			hi := float32(int(b>>4)&0xF) - 8
			lo := float32(int(b)&0xF) - 8
			w[o*inDim+2*c] = hi * q.scale[o*(inDim/quantGroup)+(2*c)/quantGroup]
			w[o*inDim+2*c+1] = lo * q.scale[o*(inDim/quantGroup)+(2*c+1)/quantGroup]
		}
	}
	return w
}

func (q *quantLinear) forward(x []float32) []float32 {
	w := q.dequantize() // transient; GC'd after
	return matmulVec(x, w, q.outFeatures)
}

// quantizeInt8: (out,in) f32 → int8 codes + f32 scales (from f16 values).
func quantizeInt8(w []float32, outDim, inDim int) ([]byte, []float32) {
	q := make([]byte, outDim*inDim)
	scale := make([]float32, outDim*(inDim/quantGroup))
	for o := 0; o < outDim; o++ {
		for b := 0; b < inDim/quantGroup; b++ {
			var amax float32
			for j := 0; j < quantGroup; j++ {
				if a := abs32(w[o*inDim+b*quantGroup+j]); a > amax {
					amax = a
				}
			}
			s := amax / 127
			if s < 1e-8 {
				s = 1e-8
			}
			scale[o*(inDim/quantGroup)+b] = s
			for j := 0; j < quantGroup; j++ {
				v := roundF32(w[o*inDim+b*quantGroup+j] / s)
				q[o*inDim+b*quantGroup+j] = byte(int8(clamp(v, -127, 127)))
			}
		}
	}
	return q, scale
}

func quantizeInt4(w []float32, outDim, inDim int) ([]byte, []float32) {
	q := make([]byte, outDim*inDim/2)
	scale := make([]float32, outDim*(inDim/quantGroup))
	for o := 0; o < outDim; o++ {
		for b := 0; b < inDim/quantGroup; b++ {
			var amax float32
			for j := 0; j < quantGroup; j++ {
				if a := abs32(w[o*inDim+b*quantGroup+j]); a > amax {
					amax = a
				}
			}
			s := amax / 7
			if s < 1e-8 {
				s = 1e-8
			}
			scale[o*(inDim/quantGroup)+b] = s
			for j := 0; j < quantGroup; j++ {
				v := clamp(roundF32(w[o*inDim+b*quantGroup+j]/s), -7, 7) + 8
				if j%2 == 0 {
					q[o*(inDim/2)+b*quantGroup/2+j/2] |= byte(int(v)&0xF) << 4
				} else {
					q[o*(inDim/2)+b*quantGroup/2+j/2] |= byte(int(v) & 0xF)
				}
			}
		}
	}
	return q, scale
}

// quantLinearFromDense converts a linear's weights to a quantized one.
func quantLinearFromDense(w []float32, outDim, inDim, bits int) *quantLinear {
	q := &quantLinear{bits: bits, outFeatures: outDim, inFeatures: inDim}
	if bits == 8 {
		q.qweight, q.scale = quantizeInt8(w, outDim, inDim)
	} else {
		q.qweight, q.scale = quantizeInt4(w, outDim, inDim)
	}
	return q
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func clamp(x, lo, hi float32) float32 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func roundF32(x float32) float32 { return float32(math.Round(float64(x))) }

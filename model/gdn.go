package model

import "math"

// GatedDeltaNet — the linear-attention mixer from Qwen3.5+, ported 1:1 from
// tiny-qwen. Recurrent state S (per v-head, d_k × d_v) is fixed-size fast
// weight memory; per token: forget (exp decay), surprise (delta rule),
// write, read.

type gatedDeltaNet struct {
	inProjQKV linear // keyDim*2 + valueDim out
	inProjZ   linear // valueDim out (output gate)
	inProjB   linear // nVHeads out (write strength beta)
	inProjA   linear // nVHeads out (decay gate input)
	outProj   linear // nEmbed out
	qkvQuant  *quantLinear
	zQuant    *quantLinear
	bQuant    *quantLinear
	aQuant    *quantLinear
	outQuant  *quantLinear
	convW     []float32 // depthwise conv, (kernel, convDim)
	dtBias    []float32 // nVHeads
	aLog      []float32 // nVHeads
	normW     []float32 // RMSNormGated weight, dV
	nEmbed    int
	nKHeads   int
	nVHeads   int
	dK        int
	dV        int
	convDim   int
	kernel    int
}

type gdnCache struct {
	S    [][]float32 // per v-head: dK*dV
	conv []float32   // last kernel-1 qkv channels
}

func newGdnCache(g *gatedDeltaNet) *gdnCache {
	c := &gdnCache{}
	for h := 0; h < g.nVHeads; h++ {
		c.S = append(c.S, make([]float32, g.dK*g.dV))
	}
	c.conv = make([]float32, g.convDim*(g.kernel-1))
	return c
}

func newGatedDeltaNet(cfg *Config) *gatedDeltaNet {
	g := &gatedDeltaNet{
		nKHeads: cfg.NLinearKHeads,
		nVHeads: cfg.NLinearVHeads,
		dK:      cfg.DLinearK,
		dV:      cfg.DLinearV,
		convDim: cfg.NLinearKHeads*cfg.DLinearK*2 + cfg.NLinearVHeads*cfg.DLinearV,
		kernel:  cfg.LinearConvKernel,
	}
	g.dtBias = make([]float32, g.nVHeads)
	for i := range g.dtBias {
		g.dtBias[i] = 1
	}
	g.aLog = make([]float32, g.nVHeads)
	g.normW = make([]float32, g.dV)
	return g
}

func (g *gatedDeltaNet) forward(x []float32, T int, cache *gdnCache) []float32 {
	ne := len(x) / T
	H := g.nVHeads
	r := g.nVHeads / g.nKHeads

	// per-token scalars
	beta := make([]float32, T*H) // sigmoid(in_proj_b)
	gg := make([]float32, T*H)   // -exp(A_log) * softplus(a + dt_bias)
	{
		b := proj(x, T, &g.inProjB, g.bQuant, H)
		a := proj(x, T, &g.inProjA, g.aQuant, H)
		traceCK("gdn_b", b)
		traceCK("gdn_a", a)
		for t := 0; t < T; t++ {
			for h := 0; h < H; h++ {
				beta[t*H+h] = sigmoid1(b[t*H+h])
				gg[t*H+h] = -float32(math.Exp(float64(g.aLog[h]))) *
					softplus(a[t*H+h]+g.dtBias[h])
			}
		}
	}

	traceCK("gdn_beta", beta)
	traceCK("gdn_g", gg)
	// causal depthwise conv over cached window ++ new tokens, then SiLU
	qkv := proj(x, T, &g.inProjQKV, g.qkvQuant, g.convDim) // (T, convDim)
	traceCK("gdn_qkv", qkv)
	combined := make([]float32, 0, len(cache.conv)+len(qkv))
	combined = append(combined, cache.conv...)
	combined = append(combined, qkv...)
	traceCK("gdn_conv_presilu_window", combined)
	// new cache window: last kernel-1 channels (all from the new tokens)
	copy(cache.conv, combined[len(combined)-g.convDim*(g.kernel-1):])
	convRaw := make([]float32, T*g.convDim)
	convOut := make([]float32, T*g.convDim)
	for t := 0; t < T; t++ {
		for ch := 0; ch < g.convDim; ch++ {
			var s float32
			// causal window: combined already starts with the K-1 token
			// window (zeros on the first call = the causal left pad), so
			// torch's padded[t+j] maps directly to combined[t+j]
			for j := 0; j < g.kernel; j++ {
				idx := t + j
				var val float32
				if idx >= len(combined)/g.convDim {
					val = 0
				} else {
					val = combined[idx*g.convDim+ch]
				}
				// torch conv1d weight: (out_ch, 1, kernel) → W[ch,0,j]
				s += g.convW[ch*g.kernel+j] * val
			}
			convRaw[t*g.convDim+ch] = s
			convOut[t*g.convDim+ch] = s / (1 + float32(math.Exp(-float64(s))))
		}
	}

	traceCK("gdn_conv", convRaw)
	// split + head views; l2norm q,k with per-v-head repeat
	q := make([]float32, T*H*g.dK)
	k := make([]float32, T*H*g.dK)
	v := make([]float32, T*H*g.dV)
	kd := g.nKHeads * g.dK
	for t := 0; t < T; t++ {
		for kh := 0; kh < g.nKHeads; kh++ {
			for vh := 0; vh < r; vh++ {
				h := kh*r + vh
				copy(q[t*H*g.dK+h*g.dK:], convOut[t*g.convDim+kh*g.dK:][:g.dK])
				copy(k[t*H*g.dK+h*g.dK:], convOut[t*g.convDim+kd+kh*g.dK:][:g.dK])
			}
		}
		for h := 0; h < H; h++ {
			copy(v[t*H*g.dV+h*g.dV:], convOut[t*g.convDim+2*kd+h*g.dV:][:g.dV])
		}
	}
	scale := float32(1 / math.Sqrt(float64(g.dK)))
	for t := 0; t < T; t++ {
		for h := 0; h < H; h++ {
			l2norm(q[t*H*g.dK+h*g.dK : t*H*g.dK+(h+1)*g.dK])
			for i := 0; i < g.dK; i++ {
				q[t*H*g.dK+h*g.dK+i] *= scale
			}
			l2norm(k[t*H*g.dK+h*g.dK : t*H*g.dK+(h+1)*g.dK])
		}
	}

	// the recurrence
	out := make([]float32, T*H*g.dV)
	for t := 0; t < T; t++ {
		for h := 0; h < H; h++ {
			S := cache.S[h]
			dec := float32(math.Exp(float64(gg[t*H+h])))
			b := beta[t*H+h]
			kt := k[t*H*g.dK+h*g.dK : t*H*g.dK+(h+1)*g.dK][:g.dK]
			qt := q[t*H*g.dK+h*g.dK : t*H*g.dK+h*g.dK+g.dK]
			vt := v[t*H*g.dV+h*g.dV : t*H*g.dV+h*g.dV+g.dV]
			ot := out[t*H*g.dV+h*g.dV : t*H*g.dV+h*g.dV+g.dV]

			// S *= exp(g)
			for i := range S {
				S[i] *= dec
			}
			// read: u[h] = S[h] @ k
			u := make([]float32, g.dV)
			for kk := 0; kk < g.dK; kk++ {
				kv := kt[kk]
				row := S[kk*g.dV : kk*g.dV+g.dV]
				for vv := 0; vv < g.dV; vv++ {
					u[vv] += kv * row[vv]
				}
			}
			// delta = beta * (v - u); S += outer(k, delta)
			delta := make([]float32, g.dV)
			for vv := 0; vv < g.dV; vv++ {
				delta[vv] = b * (vt[vv] - u[vv])
			}
			for kk := 0; kk < g.dK; kk++ {
				kv := kt[kk]
				row := S[kk*g.dV : kk*g.dV+g.dV]
				for vv := 0; vv < g.dV; vv++ {
					row[vv] += kv * delta[vv]
				}
			}
			// read: out = S @ q
			for kk := 0; kk < g.dK; kk++ {
				qv := qt[kk]
				row := S[kk*g.dV : kk*g.dV+g.dV]
				for vv := 0; vv < g.dV; vv++ {
					ot[vv] += qv * row[vv]
				}
			}
		}
	}

	traceCK("gdn_recur", out)
	// gated output norm + project
	y := make([]float32, T*ne)
	z := proj(x, T, &g.inProjZ, g.zQuant, H*g.dV)
	traceCK("gdn_z", z)
	eps := float32(1e-6)
	for t := 0; t < T; t++ {
		for h := 0; h < H; h++ {
			o := out[t*H*g.dV+h*g.dV : t*H*g.dV+(h+1)*g.dV]
			zt := z[t*H*g.dV+h*g.dV : t*H*g.dV+(h+1)*g.dV]
			rmsNormGatedApply(o, g.normW, zt, eps)
		}
		copy(y[t*ne:(t+1)*ne], proj(out[t*H*g.dV:(t+1)*H*g.dV], 1, &g.outProj, g.outQuant, ne))
	}
	traceCK("gdn_norm", out)
	traceCK("gdn_out", y)
	return y
}

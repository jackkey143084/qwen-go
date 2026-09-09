package model

import (
	"fmt"
	"math"
	"runtime"
)

// ---------------------------------------------------------------------------
// Minimal tensor ops over flat []float32, row-major. Pure Go, no cgo.
// ---------------------------------------------------------------------------

// matmul computes y = x @ W^T; x is (T, in), W is (out, in), y is (T, out).
// Parallelized across output rows with goroutines — decode spends all its
// time here, and FreeBSD's SCHED_ULE scheduler scales fine on small worker
// counts.
func matmul(x []float32, w []float32, T, in, out int) []float32 {
	y := make([]float32, T*out)
	const minWork = 1 << 16
	if T*out < minWork {
		matmulRange(x, w, y, T, in, out, 0, T)
		return y
	}
	workers := runtimeNumWorkers()
	chunk := (T + workers - 1) / workers
	done := make(chan struct{}, workers)
	spawned := 0
	for i := 0; i < workers; i++ {
		lo := i * chunk
		hi := lo + chunk
		if hi > T {
			hi = T
		}
		if lo >= hi {
			break
		}
		spawned++
		go func(lo, hi int) {
			defer func() { done <- struct{}{} }()
			matmulRange(x, w, y, T, in, out, lo, hi)
		}(lo, hi)
	}
	for i := 0; i < spawned; i++ {
		<-done
	}
	return y
}

// runtimeNumWorkers picks a modest parallelism level for matmuls.
func runtimeNumWorkers() int {
	n := runtime.GOMAXPROCS(0)
	if n > 8 {
		n = 8
	}
	if n < 1 {
		n = 1
	}
	return n
}

func matmulRange(x, w, y []float32, T, in, out, tLo, tHi int) {
	for t := tLo; t < tHi; t++ {
		xt := x[t*in : (t+1)*in]
		yt := y[t*out : (t+1)*out]
		for o := 0; o < out; o++ {
			wr := w[o*in : (o+1)*in]
			var s float32
			for k, xv := range xt {
				s += xv * wr[k]
			}
			yt[o] = s
		}
	}
}

// matmulVec computes y = x @ W^T for one vector.
func matmulVec(x, w []float32, out int) []float32 {
	in := len(x)
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		wr := w[o*in : (o+1)*in]
		var s float32
		for k, xv := range x {
			s += xv * wr[k]
		}
		y[o] = s
	}
	return y
}

func silu(x []float32) {
	for i := range x {
		x[i] = x[i] / (1 + float32(math.Exp(-float64(x[i]))))
	}
}

func sigmoid1(x float32) float32 { return 1 / (1 + float32(math.Exp(-float64(x)))) }

func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(float64(math.Exp(float64(x)))))
}

func softmax(v []float32) {
	mx := v[0]
	for _, x := range v[1:] {
		if x > mx {
			mx = x
		}
	}
	var sum float32
	for i := range v {
		v[i] = float32(math.Exp(float64(v[i] - mx)))
		sum += v[i]
	}
	for i := range v {
		v[i] /= sum
	}
}

func l2norm(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(s+1e-6))
	for i := range v {
		v[i] *= inv
	}
}

// rmsNormApply: x = x/rms(x) * w  (classic RMSNorm).
func rmsNormApply(x, w []float32, eps float32) {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(s/float64(len(x))+float64(eps)))
	for i := range x {
		x[i] = x[i] * inv * w[i]
	}
}

// gemmaRMSNormApply: x = x/rms(x) * (1+w)  (Qwen3.5 variant).
func gemmaRMSNormApply(x, w []float32, eps float32) {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(s/float64(len(x))+float64(eps)))
	for i := range x {
		x[i] = x[i] * inv * (1 + w[i])
	}
}

// rmsNormGatedApply: (w * x/rms(x)) * silu(gate).
func rmsNormGatedApply(x, w, gate []float32, eps float32) {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(s/float64(len(x))+float64(eps)))
	for i := range x {
		x[i] = x[i] * inv * w[i] * (gate[i] / (1 + float32(math.Exp(-float64(gate[i])))))
	}
}

func checkFinite(x []float32, what string) error {
	for i, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%s: non-finite value at index %d", what, i)
		}
	}
	return nil
}

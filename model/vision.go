package model

// Pure-Go port of tiny_qwen/vision.py: Qwen3.5's ViT-style vision tower.
// Each image is encoded independently (the Python reference batches images
// behind a cu_seqlens window mask; per-image encoding is mathematically
// identical since attention never crosses windows).

import (
	"fmt"
	"math"
	"os"

	"tinyqwengo/safetensors"
)

type VisionConfig struct {
	NEmbed    int // hidden_size 768
	NLayer    int // depth 12
	NHeads    int // 12
	NOutput   int // out_hidden_size 1024
	NMlp      int // 3072
	NumPos    int // 2304 (48x48 learnable pos-embed grid)
	InCh      int
	TempPatch int
	Patch     int
	Merge     int
}

func intOf(m map[string]any, key string, def int) int {
	if v, ok := m[key].(float64); ok {
		return int(v)
	}
	return def
}

func visionConfigFromJSON(m map[string]any) *VisionConfig {
	vc := &VisionConfig{
		NEmbed:    intOf(m, "hidden_size", 768),
		NLayer:    intOf(m, "depth", 12),
		NHeads:    intOf(m, "num_heads", 12),
		NOutput:   intOf(m, "out_hidden_size", 1024),
		NMlp:      intOf(m, "intermediate_size", 3072),
		NumPos:    intOf(m, "num_position_embeddings", 2304),
		InCh:      intOf(m, "in_channels", 3),
		TempPatch: intOf(m, "temporal_patch_size", 2),
		Patch:     intOf(m, "patch_size", 16),
		Merge:     intOf(m, "spatial_merge_size", 2),
	}
	return vc
}

// visionTower holds bf16 raw weights (zero-copy into the shard, like the LLM)
// plus small f32 tensors (layernorms, pos embed).
type visionTower struct {
	cfg *VisionConfig
	m   *model

	headDim int // nEmbed / nHeads = 64

	// patch_embed.proj: (nEmbed, inCh*tempPatch*patch*patch) as bf16 rows
	projW []byte
	projB []float32

	posEmbed []float32 // (numPos, nEmbed) f32

	blocks []visionBlock

	mergerNormW, mergerNormB []float32
	mergerFC1W               []byte
	mergerFC1B               []float32
	mergerFC2W               []byte
	mergerFC2B               []float32
}

type visionBlock struct {
	norm1W, norm1B []float32
	norm2W, norm2B []float32
	qkvW           []byte
	qkvB           []float32
	projW          []byte
	projB          []float32
	fc1W           []byte
	fc1B           []float32
	fc2W           []byte
	fc2B           []float32
}

// newVisionTower extracts vision weights from the parsed shard files.
// Tensors are named model.visual.*.
func newVisionTower(m *model, files []*safetensors.File) (*visionTower, error) {
	vc := visionConfigFromJSON(m.cfg.VisionConfigMap)
	v := &visionTower{cfg: vc, m: m, headDim: vc.NEmbed / vc.NHeads}

	find := func(name string) (*safetensors.File, bool) {
		for _, f := range files {
			if _, ok := f.Info("model.visual." + name); ok {
				return f, true
			}
		}
		return nil, false
	}
	raw := func(name string) ([]byte, error) {
		f, ok := find(name)
		if !ok {
			return nil, fmt.Errorf("vision: missing tensor model.visual.%s", name)
		}
		b, _, dt, err := f.Raw("model.visual." + name)
		if err != nil {
			return nil, err
		}
		if dt != safetensors.BF16 {
			return nil, fmt.Errorf("vision: %s: dtype %s, want BF16", name, dt)
		}
		return b, nil
	}
	f32t := func(name string) ([]float32, error) {
		f, ok := find(name)
		if !ok {
			return nil, fmt.Errorf("vision: missing tensor model.visual.%s", name)
		}
		x, _, err := f.F32("model.visual." + name)
		return x, err
	}
	var err error
	if v.projW, err = raw("patch_embed.proj.weight"); err != nil {
		return nil, err
	}
	if v.projB, err = f32t("patch_embed.proj.bias"); err != nil {
		return nil, err
	}
	if v.posEmbed, err = f32t("pos_embed.weight"); err != nil {
		return nil, err
	}
	for i := 0; i < vc.NLayer; i++ {
		b := visionBlock{}
		p := fmt.Sprintf("blocks.%d.", i)
		if b.norm1W, err = f32t(p + "norm1.weight"); err != nil {
			return nil, err
		}
		if b.norm1B, err = f32t(p + "norm1.bias"); err != nil {
			return nil, err
		}
		if b.norm2W, err = f32t(p + "norm2.weight"); err != nil {
			return nil, err
		}
		if b.norm2B, err = f32t(p + "norm2.bias"); err != nil {
			return nil, err
		}
		if b.qkvW, err = raw(p + "attn.qkv.weight"); err != nil {
			return nil, err
		}
		if b.qkvB, err = f32t(p + "attn.qkv.bias"); err != nil {
			return nil, err
		}
		if b.projW, err = raw(p + "attn.proj.weight"); err != nil {
			return nil, err
		}
		if b.projB, err = f32t(p + "attn.proj.bias"); err != nil {
			return nil, err
		}
		if b.fc1W, err = raw(p + "mlp.linear_fc1.weight"); err != nil {
			return nil, err
		}
		if b.fc1B, err = f32t(p + "mlp.linear_fc1.bias"); err != nil {
			return nil, err
		}
		if b.fc2W, err = raw(p + "mlp.linear_fc2.weight"); err != nil {
			return nil, err
		}
		if b.fc2B, err = f32t(p + "mlp.linear_fc2.bias"); err != nil {
			return nil, err
		}
		v.blocks = append(v.blocks, b)
	}
	if v.mergerNormW, err = f32t("merger.norm.weight"); err != nil {
		return nil, err
	}
	if v.mergerNormB, err = f32t("merger.norm.bias"); err != nil {
		return nil, err
	}
	if v.mergerFC1W, err = raw("merger.linear_fc1.weight"); err != nil {
		return nil, err
	}
	if v.mergerFC1B, err = f32t("merger.linear_fc1.bias"); err != nil {
		return nil, err
	}
	if v.mergerFC2W, err = raw("merger.linear_fc2.weight"); err != nil {
		return nil, err
	}
	if v.mergerFC2B, err = f32t("merger.linear_fc2.bias"); err != nil {
		return nil, err
	}
	return v, nil
}

// encode runs one image's patches (N, 1536) through the tower and returns
// (N/4, nOutput) merged features.
func (v *visionTower) encode(pixels []float32, g ImageGrid) ([]float32, error) {
	vc := v.cfg
	N := g.T * g.H * g.W
	if len(pixels) != N*PixelsPerPatch {
		return nil, fmt.Errorf("vision: got %d patch values, want %d", len(pixels), N*PixelsPerPatch)
	}

	// patch embed: x = pixels @ projW^T + projB. projW is (nEmbed, 1536) bf16.
	x := make([]float32, N*vc.NEmbed)
	projIn := PixelsPerPatch
	for n := 0; n < N; n++ {
		matvecRange(pixels[n*projIn:(n+1)*projIn], v.projW, x[n*vc.NEmbed:(n+1)*vc.NEmbed], projIn, vc.NEmbed, 0, vc.NEmbed)
		for c := 0; c < vc.NEmbed; c++ {
			x[n*vc.NEmbed+c] += v.projB[c]
		}
	}
	traceCK("vision.patch_embed", x)

	// learnable pos embed, bilinearly interpolated to the (h, w) patch grid,
	// then reordered (hBlock, wBlock, mH, mW) like the patchify step.
	pos := v.interpolatePosEmbed(g)
	traceCK("vision.posonly", pos)
	if debugToken0 {
		dumpF32("/tmp/go_pos.f32", pos[:N*vc.NEmbed])
		dumpF32("/tmp/go_hidden.f32", x[:N*vc.NEmbed])
	}
	for i := range x {
		x[i] += pos[i]
	}
	traceCK("vision.hidden", x)

	// rotary tables for this grid
	cos, sin := v.rotTables(g)

	for i := range v.blocks {
		x = v.blockForward(&v.blocks[i], i, x, N, cos, sin)
		traceCK(fmt.Sprintf("vision.block.%d", i), x)
	}

	// merger: layernorm on 768, then 2x2-merge to 3072, fc1, gelu(tanh... no:
	// PatchMerger uses exact-erf GELU), fc2 -> nOutput.
	nOut := N / (vc.Merge * vc.Merge)
	merged := make([]float32, nOut*v.NMerge())
	for n := 0; n < nOut; n++ {
		// source tokens: n-th group of 4 consecutive patches, each (mH, mW)
		// in patchify order → features concatenated in the same order
		for q := 0; q < vc.Merge*vc.Merge; q++ {
			tok := n*vc.Merge*vc.Merge + q
			ln := layerNorm(x[tok*vc.NEmbed:(tok+1)*vc.NEmbed], v.mergerNormW, v.mergerNormB, 1e-6)
			copy(merged[n*v.NMerge()+q*vc.NEmbed:], ln)
		}
	}
	traceCK("vision.merger.norm", merged)

	h1 := make([]float32, nOut*v.NMerge())
	mat2DRaw(v.mergerFC1W, v.mergerFC1B, merged, h1, nOut, v.NMerge(), v.NMerge())
	for i := range h1 {
		h1[i] = geluErf(h1[i])
	}
	traceCK("vision.merger.fc1", h1)

	out := make([]float32, nOut*vc.NOutput)
	mat2DRaw(v.mergerFC2W, v.mergerFC2B, h1, out, nOut, v.NMerge(), vc.NOutput)
	traceCK("vision.merger.out", out)
	return out, nil
}

// NMerge is merger input width: nEmbed * merge^2 = 3072.
func (v *visionTower) NMerge() int { return v.cfg.NEmbed * v.cfg.Merge * v.cfg.Merge }

// interpolatePosEmbed mirrors fast_pos_embed_interpolate: bilinear resample
// of the (48, 48) learned grid to (h, w), then the merge permutation.
func (v *visionTower) interpolatePosEmbed(g ImageGrid) []float32 {
	vc := v.cfg
	grid := int(math.Round(math.Sqrt(float64(vc.NumPos)))) // 48
	H, W := g.H, g.W
	out := make([]float32, H*W*vc.NEmbed)
	for i := 0; i < H; i++ {
		hIdx := linspaceIdx(grid, H, i) // (floor, frac)
		for j := 0; j < W; j++ {
			wIdx := linspaceIdx(grid, W, j)
			dst := out[(i*W+j)*vc.NEmbed : (i*W+j+1)*vc.NEmbed]
			fh, fw := hIdx.fw, wIdx.fw
			for c := 0; c < vc.NEmbed; c++ {
				tl := v.posEmbed[(hIdx.lo*grid+wIdx.lo)*vc.NEmbed+c]
				tr := v.posEmbed[(hIdx.lo*grid+wIdx.hi)*vc.NEmbed+c]
				bl := v.posEmbed[(hIdx.hi*grid+wIdx.lo)*vc.NEmbed+c]
				br := v.posEmbed[(hIdx.hi*grid+wIdx.hi)*vc.NEmbed+c]
				dst[c] = float32(
					float64(tl)*(1-fh)*(1-fw) +
						float64(tr)*(1-fh)*fw +
						float64(bl)*fh*(1-fw) +
						float64(br)*fh*fw)
			}
		}
	}
	// reorder from (H, W) raster to (H/m, W/m, m, m)
	reordered := make([]float32, H*W*vc.NEmbed)
	hB, wB := H/vc.Merge, W/vc.Merge
	for hb := 0; hb < hB; hb++ {
		for wb := 0; wb < wB; wb++ {
			for mh := 0; mh < vc.Merge; mh++ {
				for mw := 0; mw < vc.Merge; mw++ {
					src := (hb*vc.Merge+mh)*W + wb*vc.Merge + mw
					dst := (hb*wB+wb)*vc.Merge*vc.Merge + mh*vc.Merge + mw
					copy(reordered[dst*vc.NEmbed:(dst+1)*vc.NEmbed], out[src*vc.NEmbed:(src+1)*vc.NEmbed])
				}
			}
		}
	}
	return reordered
}

type linIdx struct {
	lo, hi int
	fw     float64
}

func linspaceIdx(grid, n, i int) linIdx {
	// torch.linspace(0, grid-1, n) sampled at i
	v := float64(i) * float64(grid-1) / float64(n-1)
	lo := int(math.Floor(v))
	hi := lo + 1
	if hi > grid-1 {
		hi = grid - 1
	}
	return linIdx{lo: lo, hi: hi, fw: v - float64(lo)}
}

// rotTables builds per-token cos/sin (N, headDim*2 dup) for the vision
// rotary: per position (ph, pw), freqs = [invF*ph (16), invF*pw (16)]
// duplicated to headDim.
func (v *visionTower) rotTables(g ImageGrid) ([]float32, []float32) {
	hd := v.headDim
	half := hd / 2 // 32
	invF := make([]float64, half/2)
	for i := range invF {
		invF[i] = 1.0 / math.Pow(10000.0, float64(2*i)/float64(half))
	}
	N := g.H * g.W
	cos := make([]float32, N*hd)
	sin := make([]float32, N*hd)
	n := 0
	// token order is the merge order (hBlock, wBlock, mH, mW) — the same
	// permutation the patchify step and pos-embed reorder use.
	wB := g.W / spatialMergeSize
	for hb := 0; hb < g.H/spatialMergeSize; hb++ {
		for wb := 0; wb < wB; wb++ {
			for mh := 0; mh < spatialMergeSize; mh++ {
				for mw := 0; mw < spatialMergeSize; mw++ {
					i := hb*spatialMergeSize + mh
					j := wb*spatialMergeSize + mw
					v.rotRow(cos, sin, n, i, j, hd)
					n++
				}
			}
		}
	}
	return cos, sin
}

// rotRow fills token n's cos/sin from patch position (i, j).
func (v *visionTower) rotRow(cos, sin []float32, n, i, j, hd int) {
	half := hd / 2
	invF := make([]float64, half/2)
	for p := range invF {
		invF[p] = 1.0 / math.Pow(10000.0, float64(2*p)/float64(half))
	}
	for p := 0; p < half; p++ {
		var f float64
		if p < half/2 {
			f = float64(i) * invF[p]
		} else {
			f = float64(j) * invF[p-half/2]
		}
		c, s := math.Cos(f), math.Sin(f)
		cos[n*hd+p] = float32(c)
		cos[n*hd+half+p] = float32(c)
		sin[n*hd+p] = float32(s)
		sin[n*hd+half+p] = float32(s)
	}
}

func (v *visionTower) blockForward(b *visionBlock, blockIdx int, x []float32, N int, cos, sin []float32) []float32 {
	vc := v.cfg
	hd := v.headDim

	// x + attn(norm1(x))
	h := make([]float32, N*vc.NEmbed)
	for n := 0; n < N; n++ {
		copy(h[n*vc.NEmbed:(n+1)*vc.NEmbed], layerNorm(x[n*vc.NEmbed:(n+1)*vc.NEmbed], b.norm1W, b.norm1B, 1e-6))
	}
	// qkv: (N, 3*nEmbed); rows laid out (3, heads, headDim)
	qkv := make([]float32, N*3*vc.NEmbed)
	mat2DRaw(b.qkvW, b.qkvB, h, qkv, N, vc.NEmbed, 3*vc.NEmbed)
	traceCK("vision.qkv", qkv)

	attn := make([]float32, N*vc.NEmbed)
	for hIdx := 0; hIdx < vc.NHeads; hIdx++ {
		// gather q, k, v for this head; apply rotary
		type vec = []float32
		qs := make([]vec, N)
		ks := make([]vec, N)
		vs := make([]vec, N)
		for n := 0; n < N; n++ {
			base := qkv[n*3*vc.NEmbed:]
			q := base[hIdx*hd : (hIdx+1)*hd]
			k := base[vc.NEmbed+hIdx*hd : vc.NEmbed+(hIdx+1)*hd]
			vv := base[2*vc.NEmbed+hIdx*hd : 2*vc.NEmbed+(hIdx+1)*hd]
			applyVisionRotary(q, cos, sin, n, hd)
			applyVisionRotary(k, cos, sin, n, hd)
			if hIdx == 0 && n < 2 {
				traceCK(fmt.Sprintf("vision.qrot.h0.n%d", n), q)
			}
			qs[n], ks[n], vs[n] = q, k, vv
		}
		scale := 1.0 / math.Sqrt(float64(hd))
		for n := 0; n < N; n++ {
			scores := make([]float64, N)
			maxs := math.Inf(-1)
			for m := 0; m < N; m++ {
				var d float64
				for d2 := 0; d2 < hd; d2++ {
					d += float64(qs[n][d2]) * float64(ks[m][d2])
				}
				scores[m] = d * scale
				if scores[m] > maxs {
					maxs = scores[m]
				}
			}
			sum := 0.0
			for m := range scores {
				scores[m] = math.Exp(scores[m] - maxs)
				sum += scores[m]
			}
			for d2 := 0; d2 < hd; d2++ {
				var acc float64
				for m := 0; m < N; m++ {
					acc += scores[m] * float64(vs[m][d2])
				}
				attn[n*vc.NEmbed+hIdx*hd+d2] = float32(acc / sum)
			}
		}
	}
	traceCK("vision.attn0", attn)
	proj := make([]float32, N*vc.NEmbed)
	mat2DRaw(b.projW, b.projB, attn, proj, N, vc.NEmbed, vc.NEmbed)
	for i := range x {
		x[i] += proj[i]
	}
	traceCK("vision.after_attn", x)
	if debugToken0 {
		fmt.Printf("GO a0[0:8]  %s\n", fmtSlice(x[0:8]))
		fmt.Printf("GO a0 var %.8f mean %.8f\n", varianceOf(x[0:vc.NEmbed]), meanOf(x[0:vc.NEmbed]))

	}

	// x + mlp(norm2(x))  — gelu_pytorch_tanh
	for n := 0; n < N; n++ {
		copy(h[n*vc.NEmbed:(n+1)*vc.NEmbed], layerNorm(x[n*vc.NEmbed:(n+1)*vc.NEmbed], b.norm2W, b.norm2B, 1e-6))
	}
	traceCK("vision.norm2", h)
	if debugToken0 {
		fmt.Printf("GO n20[0:8] %s\n", fmtSlice(h[0:8]))
		if blockIdx == 0 {
			dumpF32("/tmp/go_norm2_b0.f32", h[:N*vc.NEmbed])
			dumpF32("/tmp/go_after_attn_b0.f32", x[:N*vc.NEmbed])
			dumpF32("/tmp/go_attn_preproj.f32", attn[:N*vc.NEmbed])
			dumpF32("/tmp/go_proj_out.f32", proj[:N*vc.NEmbed])
			dumpF32("/tmp/go_qkv.f32", qkv[:N*3*vc.NEmbed])
		}
	}
	fc1 := make([]float32, N*vc.NMlp)
	mat2DRaw(b.fc1W, b.fc1B, h, fc1, N, vc.NEmbed, vc.NMlp)
	if debugToken0 && blockIdx == 0 {
		dumpF32("/tmp/go_fc1_pre.f32", fc1[:N*vc.NMlp])
	}
	traceCK("vision.fc1_pre", fc1)
	for i := range fc1 {
		fc1[i] = geluTanh(fc1[i])
	}
	traceCK("vision.fc1_gelu", fc1)
	if debugToken0 && blockIdx == 0 {
		dumpF32("/tmp/go_fc1_gelu.f32", fc1[:N*vc.NMlp])
	}
	fc2 := make([]float32, N*vc.NEmbed)
	mat2DRaw(b.fc2W, b.fc2B, fc1, fc2, N, vc.NMlp, vc.NEmbed)
	traceCK("vision.fc2", fc2)
	if debugToken0 && blockIdx == 0 {
		dumpF32("/tmp/go_mlp_out.f32", fc2[:N*vc.NEmbed])
	}
	for i := range x {
		x[i] += fc2[i]
	}
	return x
}

func applyVisionRotary(q []float32, cos, sin []float32, n, hd int) {
	ct := cos[n*hd : (n+1)*hd]
	st := sin[n*hd : (n+1)*hd]
	half := hd / 2
	for i := 0; i < half; i++ {
		x1 := q[i]
		x2 := q[half+i]
		q[i] = x1*ct[i] - x2*st[i]
		q[half+i] = x1*st[half+i] + x2*ct[half+i]
	}
}

// mat2DRaw computes Y = X @ W^T + bias with W bf16 rows (T rows of x).
func mat2DRaw(w []byte, bias []float32, x, y []float32, T, in, out int) {
	for t := 0; t < T; t++ {
		matvecRange(x[t*in:(t+1)*in], w, y[t*out:(t+1)*out], in, out, 0, out)
		if bias != nil {
			for o := 0; o < out; o++ {
				y[t*out+o] += bias[o]
			}
		}
	}
}

func geluErf(v float32) float32 {
	return 0.5 * v * (1 + float32(math.Erf(float64(v)/math.Sqrt2)))
}

func geluTanh(v float32) float32 {
	vf := float64(v)
	return float32(0.5 * vf * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(vf+0.044715*vf*vf*vf))))
}

func layerNorm(x, w, b []float32, eps float64) []float32 {
	var mean, varr float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	for _, v := range x {
		d := float64(v) - mean
		varr += d * d
	}
	varr /= float64(len(x))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = (v - float32(mean)) / float32(math.Sqrt(varr+eps)) * w[i]
		if b != nil {
			out[i] += b[i]
		}
	}
	return out
}

// mropePositions ports Processor.get_position_ids: text tokens advance all
// three sections together; each image's token block holds section 0 at the
// current text position and advances sections 1/2 over the (h, w) patch
// grid. After the block, the text position advances by exactly 1.
func mropePositions(inputIDs []int, grids []ImageGrid, imageTokenID int) ([3][]int, error) {
	T := len(inputIDs)
	out := [3][]int{
		make([]int, T),
		make([]int, T),
		make([]int, T),
	}
	textIdx, img, seq := 0, 0, 0
	for seq < T {
		if inputIDs[seq] != imageTokenID {
			out[0][seq], out[1][seq], out[2][seq] = textIdx, textIdx, textIdx
			textIdx++
			seq++
			continue
		}
		if img >= len(grids) {
			return out, fmt.Errorf("mrope: more image tokens than images")
		}
		g := grids[img]
		hImg, wImg := g.H/g.cfg(), g.W/g.cfg()
		tokCount := hImg * wImg
		for off := 0; off < tokCount; off++ {
			hPos := off / wImg
			wPos := off % wImg
			out[0][seq+off] = textIdx
			out[1][seq+off] = textIdx + hPos
			out[2][seq+off] = textIdx + wPos
		}
		textIdx++
		img++
		seq += tokCount
	}
	return out, nil
}

func (g ImageGrid) cfg() int { return spatialMergeSize }

var debugToken0 = os.Getenv("QWENGO_DEBUG0") == "1"

func fmtSlice(v []float32) string {
	s := ""
	for _, x := range v {
		s += fmt.Sprintf("%.4f ", x)
	}
	return s
}

func varianceOf(v []float32) float64 {
	var m, q float64
	for _, x := range v {
		m += float64(x)
	}
	m /= float64(len(v))
	for _, x := range v {
		d := float64(x) - m
		q += d * d
	}
	return q / float64(len(v))
}

func meanOf(v []float32) float64 {
	var m float64
	for _, x := range v {
		m += float64(x)
	}
	return m / float64(len(v))
}

// LayerNormExport exposes layerNorm for parity tests.
func LayerNormExport(x, w, b []float32, eps float64) []float32 { return layerNorm(x, w, b, eps) }

// F32FromBits rebuilds a float32 from IEEE bits (test helper).
func F32FromBits(bits uint32) float32 { return math.Float32frombits(bits) }

func dumpF32(path string, v []float32) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		b := math.Float32bits(x)
		buf[i*4] = byte(b)
		buf[i*4+1] = byte(b >> 8)
		buf[i*4+2] = byte(b >> 16)
		buf[i*4+3] = byte(b >> 24)
	}
	if _, err := f.Write(buf); err != nil {
		panic(err)
	}
}

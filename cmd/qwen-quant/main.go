// qwen-quant: quantize a dense (BF16/F32) tiny-qwen checkpoint directory into
// a quantize.py-style directory that qwen-go's loader already understands
// (detected by quant.json: *.qweight + *.scale tensors for the packed
// linears, everything else passed through in its original dtype).
//
// Usage: go run ./cmd/qwen-quant -src <dir> -out <dir> [-bits 8|4]
//
// This is the Phase-2 "quantize mode" from ROADMAP.md, in Go, zero deps.
//
// Memory: the input shard is mmap'd (pages are kernel-reclaimable) and the
// output is streamed tensor-by-tensor, so peak heap stays around one tensor —
// comfortably inside a 3GB cgroup for a 1.6GB bf16 0.8B checkpoint.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"tinyqwengo/model"
	"tinyqwengo/safetensors"
)

const quantGroup = 32 // same group size the loader/quantizer use

// plan describes one tensor to emit in the output file.
type plan struct {
	name    string          // flat output key
	shape   []int
	dtype   safetensors.Dtype
	payloadLen int         // raw bytes to write
	shard   string          // source shard path
	srcKey  string          // source key within shard
	quant   bool            // if true, f32->quantize instead of raw passthrough
	bits    int
	scaleLen int           // payloadLen of the .scale tensor (0 if not quant)
}

// quantizeBaseNames returns the set of flat tensor base names (prefix before
// ".weight") to pack into qweight/scale: the big projection linears.
func quantizeBaseNames(cfg *model.Config) map[string]bool {
	t := map[string]bool{}
	for i := 0; i < cfg.NLayer; i++ {
		// Qwen3.5 gated attention
		for _, m := range []string{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"} {
			t[fmt.Sprintf("layers.%d.%s", i, m)] = true
		}
		// GatedDeltaNet linear attention
		for _, m := range []string{
			"linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
			"linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj",
		} {
			t[fmt.Sprintf("layers.%d.%s", i, m)] = true
		}
		// dense MLP or MoE shared-expert projections
		if cfg.NExperts > 0 {
			for _, m := range []string{"mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"} {
				t[fmt.Sprintf("layers.%d.%s", i, m)] = true
			}
		} else {
			for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"} {
				t[fmt.Sprintf("layers.%d.%s", i, m)] = true
			}
		}
	}
	return t
}

// renameKey mirrors the loader's key flattening so emitted tensor names and
// quant.json line up exactly with what FromPretrained expects.
func renameKey(key string) string {
	if strings.HasPrefix(key, "mtp.") {
		return ""
	}
	for _, p := range []struct{ old, new string }{
		{"model.language_model.", ""},
		{"model.visual.", "visual."},
		{"model.", ""},
	} {
		if strings.HasPrefix(key, p.old) {
			return p.new + key[len(p.old):]
		}
	}
	return key
}

func elementBytes(dt safetensors.Dtype) int {
	switch dt {
	case safetensors.F32, safetensors.I32:
		return 4
	case safetensors.F16, safetensors.BF16:
		return 2
	case safetensors.I8, safetensors.U8:
		return 1
	case safetensors.F64, safetensors.I64:
		return 8
	case safetensors.BOOL:
		return 1
	case safetensors.F8E4M3:
		return 1
	}
	return 4
}

func main() {
	src := flag.String("src", ".", "source checkpoint dir (config.json + *.safetensors)")
	out := flag.String("out", "", "output dir (created fresh)")
	bits := flag.Int("bits", 8, "quantization bits: 8 or 4")
	flag.Parse()
	if *out == "" {
		fatal("-out is required")
	}
	if *bits != 4 && *bits != 8 {
		fatal("-bits must be 4 or 8")
	}

	cfg, err := model.ReadConfig(*src)
	if err != nil {
		fatal("read config: %v", err)
	}
	quantBase := quantizeBaseNames(cfg)

	shards, _ := filepath.Glob(filepath.Join(*src, "*.safetensors"))
	sort.Strings(shards)
	if len(shards) == 0 {
		fatal("no .safetensors shards found in %s", *src)
	}

	if err := os.MkdirAll(*out, 0o755); err != nil {
		fatal("mkdir -out: %v", err)
	}

	// ---- Pass 1: build the write plan + safetensors header. ----
	var plans []plan
	quantized := []string{} // flat ".weight" source names we packed

	for _, shard := range shards {
		f, closer, err := model.OpenShard(shard)
		if err != nil {
			fatal("open %s: %v", shard, err)
		}
		keys := f.Keys()
		sort.Strings(keys)
		for _, key := range keys {
			flat := renameKey(key)
			if flat == "" || strings.HasPrefix(flat, "visual.") {
				continue // mtp + embedded-vision skipped (loader does too)
			}
			info, ok := f.Info(key)
			if !ok {
				continue
			}
			n := 1
			for _, d := range info.Shape {
				n *= d
			}
			rawLen := n * elementBytes(info.Dtype)

			if !strings.HasSuffix(flat, ".weight") {
				plans = append(plans, plan{
					name: flat, shape: info.Shape, dtype: info.Dtype,
					payloadLen: rawLen, shard: shard, srcKey: key,
				})
				continue
			}
			base := strings.TrimSuffix(flat, ".weight")
			if !quantBase[base] {
				plans = append(plans, plan{
					name: flat, shape: info.Shape, dtype: info.Dtype,
					payloadLen: rawLen, shard: shard, srcKey: key,
				})
				continue
			}
			if len(info.Shape) != 2 {
				fatal("%s: quantize target wants 2D, got %v", flat, info.Shape)
			}
			outDim, inDim := info.Shape[0], info.Shape[1]
			if inDim%quantGroup != 0 {
				// Not multiple of group size: fall back to passthrough rather
				// than emit a malformed quantized tensor.
				fmt.Fprintf(os.Stderr, "skip quant %s (inDim %d %% %d != 0)\n", flat, inDim, quantGroup)
				plans = append(plans, plan{
					name: flat, shape: info.Shape, dtype: info.Dtype,
					payloadLen: rawLen, shard: shard, srcKey: key,
				})
				continue
			}
			qwLen := outDim * inDim
			if *bits == 4 {
				qwLen = outDim * inDim / 2
			}
			scLen := outDim * (inDim / quantGroup) * 4
			qwShape := []int{outDim, inDim}
			if *bits == 4 {
				qwShape[1] = inDim / 2
			}
			plans = append(plans,
				plan{
					name: base + ".qweight", shape: qwShape, dtype: safetensors.I8,
					payloadLen: qwLen, shard: shard, srcKey: key, quant: true, bits: *bits,
				},
				plan{
					name: base + ".scale", shape: []int{outDim, inDim / quantGroup}, dtype: safetensors.F32,
					payloadLen: scLen, shard: shard, srcKey: key, quant: true, bits: *bits, scaleLen: scLen,
				},
			)
			quantized = append(quantized, flat)
		}
		closer.Close()
	}

	stPlans := make([]safetensors.Plan, len(plans))
	for i, p := range plans {
		stPlans[i] = safetensors.Plan{Name: p.name, Shape: p.shape, Dtype: p.dtype, Len: p.payloadLen}
	}
	sw, err := safetensors.CreateStreaming(filepath.Join(*out, "model.safetensors"), stPlans)
	if err != nil {
		fatal("create output: %v", err)
	}

	// ---- Pass 2: stream payloads. For quantized linears, decode source to
	// f32 in-memory, run the quantizer, and write qweight then scale. The
	// qweight and scale plans for a base are always emitted back-to-back in
	// pass 1, so we compute once at the .qweight plan and stash scale bytes. ----
	nQuant := 0
	scaleBytes := map[string][]byte{} // base -> packed f32 scale payload

	for _, shard := range shards {
		f, closer, err := model.OpenShard(shard)
		if err != nil {
			sw.Close()
			fatal("open %s: %v", shard, err)
		}
		for i := range plans {
			p := &plans[i]
			if p.shard != shard {
				continue
			}
			if !p.quant {
				raw, _, _, err := f.Raw(p.srcKey)
				if err != nil {
					sw.Close()
					closer.Close()
					fatal("read %s: %v", p.srcKey, err)
				}
				if err := sw.WriteAll(raw); err != nil {
					sw.Close()
					closer.Close()
					fatal("write %s: %v", p.name, err)
				}
				continue
			}
			if strings.HasSuffix(p.name, ".qweight") {
				w, _, err := f.F32(p.srcKey)
				if err != nil {
					sw.Close()
					closer.Close()
					fatal("f32 %s: %v", p.srcKey, err)
				}
				// real dims from the scale plan's shape (it carries them).
				base := strings.TrimSuffix(p.name, ".qweight")
				outDim, inDim := 0, 0
				// For simplicity recompute from the source tensor shape.
				if info, ok := f.Info(p.srcKey); ok && len(info.Shape) == 2 {
					outDim, inDim = info.Shape[0], info.Shape[1]
				}
				var qw []byte
				var sc []float32
				if p.bits == 8 {
					qw, sc = model.QuantizeInt8(w, outDim, inDim)
				} else {
					qw, sc = model.QuantizeInt4(w, outDim, inDim)
				}
				if err := sw.WriteAll(qw); err != nil {
					sw.Close()
					closer.Close()
					fatal("write %s: %v", p.name, err)
				}
				scaleBytes[base] = f32Bytes(sc)
				nQuant++
				if nQuant == 1 || nQuant%40 == 0 {
					fmt.Fprintf(os.Stderr, "quantized %d projections...\n", nQuant)
				}
				continue
			}
			// this is the paired .scale plan
			base := strings.TrimSuffix(p.name, ".scale")
			b, ok := scaleBytes[base]
			if !ok {
				sw.Close()
				closer.Close()
				fatal("no cached scale for %s (ordering broken?)", base)
			}
			if err := sw.WriteAll(b); err != nil {
				sw.Close()
				closer.Close()
				fatal("write %s: %v", p.name, err)
			}
			delete(scaleBytes, base)
		}
		closer.Close()
	}

	if err := sw.Close(); err != nil {
		fatal("close output: %v", err)
	}

	// ---- manifest + aux files ----
	manifest := struct {
		Bits      int      `json:"bits"`
		Quantized []string `json:"quantized"`
	}{Bits: *bits, Quantized: quantized}
	mb, _ := json.MarshalIndent(manifest, "", "  ")
	if err := os.WriteFile(filepath.Join(*out, "quant.json"), mb, 0o644); err != nil {
		fatal("write quant.json: %v", err)
	}
	for _, aux := range []string{"config.json", "tokenizer.json"} {
		if b, err := os.ReadFile(filepath.Join(*src, aux)); err == nil {
			_ = os.WriteFile(filepath.Join(*out, aux), b, 0o644)
		}
	}

	fmt.Printf("done: %d linears quantized to %d bits -> %s (model.safetensors + quant.json)\n",
		nQuant, *bits, *out)
}

func f32Bytes(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "qwen-quant: "+format+"\n", a...)
	os.Exit(1)
}

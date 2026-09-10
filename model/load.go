package model

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"tinyqwengo/safetensors"
	"tinyqwengo/tokenizer"
)

// OpenShard memory-maps a safetensors shard for callers that want the loader's
// zero-copy behavior without going through FromPretrained (e.g. cmd/qwen-quant
// reads shards directly to emit a quantized directory). The returned closer
// releases the mapping when done.
func OpenShard(path string) (*safetensors.File, io.Closer, error) {
	return openShard(path)
}

// Loading: HF-format directory of safetensors shards, or a quantize.py-style
// directory (detected by quant.json). mmap is used for shard reading on
// freebsd/linux (see mmap.go build tags).

// renameKey maps HF checkpoint names (Qwen3.5 nests under
// model.language_model / model.visual) to our flat names. Returns "" to skip.
func renameKey(key string) string {
	if strings.HasPrefix(key, "mtp.") {
		return "" // speculative-decoding head, unused
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

// Model is the exported handle for cmd use (alias of the internal type).
type Model = model

// quantManifest shape shared with load.go
type quantManifest struct {
	Bits      int      `json:"bits"`
	Quantized []string `json:"quantized"`
}

// FromPretrained loads a model directory (config.json + *.safetensors +
// tokenizer.json). Vision weights are skipped in this text-only phase.
func FromPretrained(weightsPath string) (*model, *tokenizer.Tokenizer, error) {
	if _, err := os.Stat(filepath.Join(weightsPath, "quant.json")); err == nil {
		return loadQuantized(weightsPath)
	}
	cfg, err := ReadConfig(weightsPath)
	if err != nil {
		return nil, nil, err
	}
	m := newModel(cfg)
	if err := m.alloc(nil, true); err != nil {
		return nil, nil, err
	}
	m.rawMode = true

	shards, err := filepath.Glob(filepath.Join(weightsPath, "*.safetensors"))
	if err != nil {
		return nil, nil, err
	}
	sort.Strings(shards)
	if len(shards) == 0 {
		return nil, nil, fmt.Errorf("no .safetensors shards in %s", weightsPath)
	}
	// NB: shard mmaps are intentionally kept open — raw weight slices point
	// into the mapped pages.
	var shardFiles []*safetensors.File
	for _, shard := range shards {
		f, closer, err := openShard(shard)
		if err != nil {
			return nil, nil, err
		}
		m.closers = append(m.closers, closer)
		shardFiles = append(shardFiles, f)
		if err := m.loadShards(f, shard); err != nil {
			return nil, nil, err
		}
	}

	// vision tower, if the checkpoint carries one
	visErr := error(nil)
	for _, f := range shardFiles {
		for _, k := range f.Keys() {
			if strings.HasPrefix(k, "model.visual.") {
				m.visual, visErr = newVisionTower(m, shardFiles)
				if visErr != nil {
					return nil, nil, visErr
				}
				break
			}
		}
		if m.visual != nil {
			break
		}
	}

	tok, err := tokenizer.Load(filepath.Join(weightsPath, "tokenizer.json"))
	if err != nil {
		return nil, nil, err
	}
	return m, tok, nil
}

func (m *model) loadShards(f *safetensors.File, shard string) error {
	for _, key := range f.Keys() {
		name := renameKey(key)
		if name == "" || strings.HasPrefix(name, "visual.") {
			continue // text-only phase
		}
		if err := m.loadTensor(name, f, key); err != nil {
			return fmt.Errorf("%s: %w", shard, err)
		}
	}
	return nil
}

// alloc materializes every weight buffer. quantTargets (names like
// "layers.3.self_attn.q_proj") skip dense allocation — their weights arrive
// as qweight/scale.
func (m *model) alloc(quantTargets map[string]bool, rawMode bool) error {
	if quantTargets == nil {
		quantTargets = map[string]bool{}
	}
	cfg := m.cfg
	ne := cfg.NEmbed
	m.norm = make([]float32, ne)
	if !rawMode {
		m.embed.w = make([]float32, cfg.NVocab*ne)
		if !m.tied {
			m.lmHead.w = make([]float32, cfg.NVocab*ne)
		}
	}
	for i := range m.layers {
		l := &m.layers[i]
		if l.attn != nil {
			a := l.attn
			a.nEmbed = ne
			if !rawMode {
				if !quantTargets[keyOf(i, "self_attn.q_proj")] {
					a.qProj.w = make([]float32, a.nHeads*2*a.dHead*ne)
				}
				if !quantTargets[keyOf(i, "self_attn.k_proj")] {
					a.kProj.w = make([]float32, a.nKV*a.dHead*ne)
				}
				if !quantTargets[keyOf(i, "self_attn.v_proj")] {
					a.vProj.w = make([]float32, a.nKV*a.dHead*ne)
				}
				if !quantTargets[keyOf(i, "self_attn.o_proj")] {
					a.oProj.w = make([]float32, ne*a.nHeads*a.dHead)
				}
			}
		}
		if l.gdn != nil {
			g := l.gdn
			g.nEmbed = ne
			if !rawMode {
				if !quantTargets[keyOf(i, "linear_attn.in_proj_qkv")] {
					g.inProjQKV.w = make([]float32, g.convDim*ne)
				}
				if !quantTargets[keyOf(i, "linear_attn.in_proj_z")] {
					g.inProjZ.w = make([]float32, g.nVHeads*g.dV*ne)
				}
				if !quantTargets[keyOf(i, "linear_attn.in_proj_b")] {
					g.inProjB.w = make([]float32, g.nVHeads*ne)
				}
				if !quantTargets[keyOf(i, "linear_attn.in_proj_a")] {
					g.inProjA.w = make([]float32, g.nVHeads*ne)
				}
				if !quantTargets[keyOf(i, "linear_attn.out_proj")] {
					g.outProj.w = make([]float32, ne*g.nVHeads*g.dV)
				}
			}
			g.convW = make([]float32, g.convDim*g.kernel)
		}
		if cfg.NExperts > 0 {
			moe := l.mlp.(*moeMLP)
			moe.nEmbedSet = ne
			if !quantTargets[keyOf(i, "mlp.gate")] && !rawMode {
				moe.gate.w = make([]float32, moe.nExperts*moe.nEmbed)
			}
			if moe.shared != nil {
				if !rawMode {
					if !quantTargets[keyOf(i, "mlp.shared_expert.gate_proj")] {
						moe.shared.gate.w = make([]float32, cfg.NSharedExpertMlp*ne)
					}
					if !quantTargets[keyOf(i, "mlp.shared_expert.up_proj")] {
						moe.shared.up.w = make([]float32, cfg.NSharedExpertMlp*ne)
					}
					if !quantTargets[keyOf(i, "mlp.shared_expert.down_proj")] {
						moe.shared.down.w = make([]float32, ne*cfg.NSharedExpertMlp)
					}
				}
				if !quantTargets[keyOf(i, "mlp.shared_expert_gate")] {
					moe.sharedGate = make([]float32, ne)
				}
			}
			_ = moe.gateUpW
		} else {
			d := l.mlp.(*denseMLP)
			d.nEmbed = ne
			d.nMlp = cfg.NMlp
			if !rawMode {
				if !quantTargets[keyOf(i, "mlp.gate_proj")] {
					d.gate.w = make([]float32, cfg.NMlp*ne)
				}
				if !quantTargets[keyOf(i, "mlp.up_proj")] {
					d.up.w = make([]float32, cfg.NMlp*ne)
				}
				if !quantTargets[keyOf(i, "mlp.down_proj")] {
					d.down.w = make([]float32, ne*cfg.NMlp)
				}
			}
		}
	}
	return nil
}

func keyOf(layer int, mod string) string {
	return fmt.Sprintf("layers.%d.%s", layer, mod)
}

// setLinear fills a projection from a safetensors tensor: bf16 stays raw
// (memory-dense), everything else upcasts to f32.
func setLinear(l *linear, f *safetensors.File, key string, rawMode bool) error {
	if _, ok := f.Info(key); !ok {
		return fmt.Errorf("no tensor %q", key)
	}
	raw, shape, dt, err := f.Raw(key)
	if err != nil {
		return err
	}
	if len(shape) != 2 {
		return fmt.Errorf("%q: want 2D, got %v", key, shape)
	}
	if rawMode && dt == safetensors.BF16 {
		// zero-copy: the slice points into the shard's mmap. Clean file
		// pages are kernel-reclaimable, so this beats copying 1.6GB into
		// the Go heap under a tight cgroup limit.
		l.raw = raw
		l.outDim, l.inDim = shape[0], shape[1]
		return nil
	}
	v, _, err := f.F32(key)
	if err != nil {
		return err
	}
	if len(v) != len(l.w) && l.w != nil {
		return fmt.Errorf("%q: want %d values, got %d", key, len(l.w), len(v))
	}
	if l.w != nil {
		copy(l.w, v)
	} else {
		l.w = v
	}
	return nil
}

func setF32(dst *[]float32, vals []float32) error {
	if len(vals) != len(*dst) {
		return fmt.Errorf("shape mismatch: want %d values, got %d", len(*dst), len(vals))
	}
	copy(*dst, vals)
	return nil
}

func (m *model) loadTensor(name string, f *safetensors.File, key string) error {
	cfg := m.cfg
	if name == "embed_tokens.weight" {
		return setLinear(&m.embed, f, key, true)
	}
	if name == "norm.weight" {
		v, _, err := f.F32(key)
		if err != nil {
			return err
		}
		return setF32(&m.norm, v)
	}
	if name == "lm_head.weight" {
		if m.tied {
			return nil
		}
		return setLinear(&m.lmHead, f, key, true)
	}
	if strings.HasPrefix(name, "layers.") {
		rest := name[len("layers."):]
		dot := strings.Index(rest, ".")
		if dot < 0 {
			return fmt.Errorf("bad layer key %q", name)
		}
		idx := 0
		for _, c := range rest[:dot] {
			if c < '0' || c > '9' {
				return fmt.Errorf("bad layer index in %q", name)
			}
			idx = idx*10 + int(c-'0')
		}
		if idx >= cfg.NLayer {
			return nil
		}
		l := &m.layers[idx]
		rel := rest[dot+1:]
		switch {
		case rel == "input_layernorm.weight":
			v, _, err := f.F32(key)
			if err != nil {
				return err
			}
			return setF32(&l.inputLayernorm, v)
		case rel == "post_attention_layernorm.weight":
			v, _, err := f.F32(key)
			if err != nil {
				return err
			}
			return setF32(&l.postAttentionLayernorm, v)
		case strings.HasPrefix(rel, "self_attn."):
			return loadAttn(l.attn, strings.TrimPrefix(rel, "self_attn."), f, key, m.rawMode)
		case strings.HasPrefix(rel, "linear_attn."):
			return loadGDN(l.gdn, strings.TrimPrefix(rel, "linear_attn."), f, key, m.rawMode)
		case strings.HasPrefix(rel, "mlp."):
			return loadMLP(cfg, l, strings.TrimPrefix(rel, "mlp."), f, key, m.rawMode)
		}
	}
	return nil // unknown keys (rotary buffers etc.) are fine to skip
}

func loadAttn(a *selfAttention, rel string, f *safetensors.File, key string, rawMode bool) error {
	switch rel {
	case "q_proj.weight":
		return setLinear(&a.qProj, f, key, rawMode)
	case "k_proj.weight":
		return setLinear(&a.kProj, f, key, rawMode)
	case "v_proj.weight":
		return setLinear(&a.vProj, f, key, rawMode)
	case "o_proj.weight":
		return setLinear(&a.oProj, f, key, rawMode)
	case "q_norm.weight":
		v, _, err := f.F32(key)
		if err != nil {
			return err
		}
		return setF32(&a.qNorm, v)
	case "k_norm.weight":
		v, _, err := f.F32(key)
		if err != nil {
			return err
		}
		return setF32(&a.kNorm, v)
	}
	return nil
}

func loadGDN(g *gatedDeltaNet, rel string, f *safetensors.File, key string, rawMode bool) error {
	switch rel {
	case "in_proj_qkv.weight":
		return setLinear(&g.inProjQKV, f, key, rawMode)
	case "in_proj_z.weight":
		return setLinear(&g.inProjZ, f, key, rawMode)
	case "in_proj_b.weight":
		return setLinear(&g.inProjB, f, key, rawMode)
	case "in_proj_a.weight":
		return setLinear(&g.inProjA, f, key, rawMode)
	case "out_proj.weight":
		return setLinear(&g.outProj, f, key, rawMode)
	case "conv1d.weight":
		v, _, err := f.F32(key)
		if err != nil {
			return err
		}
		return setF32(&g.convW, v)
	case "dt_bias":
		v, _, err := f.F32(key)
		if err != nil {
			return err
		}
		return setF32(&g.dtBias, v)
	case "A_log":
		v, _, err := f.F32(key)
		if err != nil {
			return err
		}
		return setF32(&g.aLog, v)
	case "norm.weight":
		v, _, err := f.F32(key)
		if err != nil {
			return err
		}
		return setF32(&g.normW, v)
	}
	return nil
}

func loadMLP(cfg *Config, l *block, rel string, f *safetensors.File, key string, rawMode bool) error {
	get := func(dst *[]float32) error {
		v, _, err := f.F32(key)
		if err != nil {
			return err
		}
		return setF32(dst, v)
	}
	lin := func(l *linear) error { return setLinear(l, f, key, rawMode) }
	if cfg.NExperts > 0 {
		moe := l.mlp.(*moeMLP)
		switch rel {
		case "gate.weight":
			return lin(&moe.gate)
		case "experts.gate_up_proj.weight":
			// stacked blob: raw bf16 copy or f32
			raw, shape, dt, err := f.Raw(key)
			if err != nil {
				return err
			}
			_ = shape
			if dt == safetensors.BF16 {
				moe.gateUpRaw = raw
				return nil
			}
			v, _, err := f.F32(key)
			if err != nil {
				return err
			}
			return setF32(&moe.gateUpW, v)
		case "experts.down_proj.weight":
			raw, _, dt, err := f.Raw(key)
			if err != nil {
				return err
			}
			if dt == safetensors.BF16 {
				moe.downRaw = raw
				return nil
			}
			v, _, err := f.F32(key)
			if err != nil {
				return err
			}
			return setF32(&moe.downW, v)
		case "shared_expert.gate_proj.weight":
			return setLinear(&moe.shared.gate, f, key, rawMode)
		case "shared_expert.up_proj.weight":
			return setLinear(&moe.shared.up, f, key, rawMode)
		case "shared_expert.down_proj.weight":
			return setLinear(&moe.shared.down, f, key, rawMode)
		case "shared_expert_gate.weight":
			return get(&moe.sharedGate)
		}
		return nil
	}
	d := l.mlp.(*denseMLP)
	switch rel {
	case "gate_proj.weight":
		return lin(&d.gate)
	case "up_proj.weight":
		return lin(&d.up)
	case "down_proj.weight":
		return lin(&d.down)
	}
	return nil
}

// loadQuantized loads a quantize.py-style directory: quantized Linears read
// (qweight, scale) tensors and dequantize just-in-time at forward time.
func loadQuantized(weightsPath string) (*model, *tokenizer.Tokenizer, error) {
	cfg, err := ReadConfig(weightsPath)
	if err != nil {
		return nil, nil, err
	}
	manifestRaw, err := os.ReadFile(filepath.Join(weightsPath, "quant.json"))
	if err != nil {
		return nil, nil, err
	}
	var manifest quantManifest
	if err := json.Unmarshal(manifestRaw, &manifest); err != nil {
		return nil, nil, err
	}

	quantTargets := map[string]bool{}
	for _, n := range manifest.Quantized {
		if r := renameKey(n); r != "" {
			quantTargets[strings.TrimSuffix(r, ".weight")] = true
		}
	}
	m := newModel(cfg)
	if err := m.alloc(quantTargets, true); err != nil {
		return nil, nil, err
	}
	m.rawMode = true

	shards, _ := filepath.Glob(filepath.Join(weightsPath, "*.safetensors"))
	sort.Strings(shards)
	for _, shard := range shards {
		f, closer, err := openShard(shard)
		if err != nil {
			return nil, nil, err
		}
		m.closers = append(m.closers, closer)
		err = m.loadShards(f, shard)
		if err == nil {
			err = m.loadQuantTensors(f, shard, &manifest)
		}
		if err != nil {
			return nil, nil, err
		}
	}

	tok, err := tokenizer.Load(filepath.Join(weightsPath, "tokenizer.json"))
	if err != nil {
		return nil, nil, err
	}
	return m, tok, nil
}

// loadQuantTensors fills quantLinear wrappers from qweight/scale tensors in
// one shard.
func (m *model) loadQuantTensors(f *safetensors.File, shard string, manifest *quantManifest) error {
	type pending struct {
		qw   []byte
		qwSz []int
		sc   []float32
	}
	pends := map[string]*pending{}
	for _, key := range f.Keys() {
		name := renameKey(key)
		if name == "" {
			continue
		}
		switch {
		case strings.HasSuffix(name, ".qweight"):
			raw, shape, _, err := f.Raw(key)
			if err != nil {
				return err
			}
			base := strings.TrimSuffix(name, ".qweight")
			p := pends[base]
			if p == nil {
				p = &pending{}
				pends[base] = p
			}
			p.qw = raw
			p.qwSz = shape
		case strings.HasSuffix(name, ".scale"):
			v, _, err := f.F32(key)
			if err != nil {
				return err
			}
			base := strings.TrimSuffix(name, ".scale")
			p := pends[base]
			if p == nil {
				p = &pending{}
				pends[base] = p
			}
			p.sc = v
		}
	}
	for base, p := range pends {
		if p.qw == nil || p.sc == nil || len(p.qwSz) != 2 {
			continue
		}
		outDim := p.qwSz[0]
		inDim := p.qwSz[1]
		if manifest.Bits == 4 {
			inDim *= 2
		}
		ql := &quantLinear{bits: manifest.Bits, outFeatures: outDim, inFeatures: inDim, qweight: p.qw, scale: p.sc}
		if err := m.installQuant(base, ql); err != nil {
			return fmt.Errorf("%s: %w", shard, err)
		}
	}
	return nil
}

// installQuant wires a quantLinear into the module graph by dotted name
// ("layers.3.self_attn.q_proj").
func (m *model) installQuant(name string, ql *quantLinear) error {
	if !strings.HasPrefix(name, "layers.") {
		return fmt.Errorf("unsupported quant target %q", name)
	}
	rest := name[len("layers."):]
	dot := strings.Index(rest, ".")
	if dot < 0 {
		return fmt.Errorf("bad quant target %q", name)
	}
	idx := 0
	for _, c := range rest[:dot] {
		if c < '0' || c > '9' {
			return fmt.Errorf("bad layer index in %q", name)
		}
		idx = idx*10 + int(c-'0')
	}
	if idx >= m.cfg.NLayer {
		return fmt.Errorf("layer %d out of range", idx)
	}
	l := &m.layers[idx]
	rel := rest[dot+1:]
	switch {
	case strings.HasPrefix(rel, "self_attn."):
		a := l.attn
		if a == nil {
			return fmt.Errorf("layer %d has no attention", idx)
		}
		switch rel[len("self_attn."):] {
		case "q_proj":
			a.qQuant = ql
		case "k_proj":
			a.kQuant = ql
		case "v_proj":
			a.vQuant = ql
		case "o_proj":
			a.oQuant = ql
		default:
			return fmt.Errorf("unknown attn proj %q", rel)
		}
		return nil
	case strings.HasPrefix(rel, "linear_attn."):
		g := l.gdn
		if g == nil {
			return fmt.Errorf("layer %d has no GDN", idx)
		}
		switch rel[len("linear_attn."):] {
		case "in_proj_qkv":
			g.qkvQuant = ql
		case "in_proj_z":
			g.zQuant = ql
		case "in_proj_b":
			g.bQuant = ql
		case "in_proj_a":
			g.aQuant = ql
		case "out_proj":
			g.outQuant = ql
		default:
			return fmt.Errorf("unknown gdn proj %q", rel)
		}
		return nil
	case strings.HasPrefix(rel, "mlp."):
		sub := rel[len("mlp."):]
		if m.cfg.NExperts > 0 {
			moe := l.mlp.(*moeMLP)
			switch sub {
			case "gate":
				moe.routerQuant = ql
			case "shared_expert.gate_proj":
				moe.shared.gateQuant = ql
			case "shared_expert.up_proj":
				moe.shared.upQuant = ql
			case "shared_expert.down_proj":
				moe.shared.downQuant = ql
			case "shared_expert_gate":
				moe.sharedGateQuant = ql
			default:
				return fmt.Errorf("unknown moe proj %q", sub)
			}
			return nil
		}
		d := l.mlp.(*denseMLP)
		switch sub {
		case "gate_proj":
			d.gateQuant = ql
		case "up_proj":
			d.upQuant = ql
		case "down_proj":
			d.downQuant = ql
		default:
			return fmt.Errorf("unknown mlp proj %q", sub)
		}
		return nil
	}
	return fmt.Errorf("unsupported quant target %q", name)
}

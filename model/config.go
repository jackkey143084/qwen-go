package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config is the text-model configuration, ported from tiny-qwen's
// ModelConfig. Fields follow HF Qwen config.json naming.
type Config struct {
	NEmbed             int     `json:"hidden_size"`
	NHeads             int     `json:"num_attention_heads"`
	NKVHeads           int     `json:"num_key_value_heads"`
	NLayer             int     `json:"num_hidden_layers"`
	NMlp               int     `json:"-"` // resolved from intermediate_size fallbacks
	NVocab             int     `json:"vocab_size"`
	TieWordEmbeddings  bool    `json:"-"`
	RopeTheta          float64 `json:"-"`
	RmsNormEps         float64 `json:"-"`
	DHead              int     `json:"head_dim"`
	NExperts           int     `json:"num_experts"`
	NExpertsPerToken   int     `json:"num_experts_per_tok"`
	NMoeMlp            int     `json:"moe_intermediate_size"`
	NSharedExpertMlp   int     `json:"shared_expert_intermediate_size"`
	LayerTypes         []string `json:"layer_types"`
	NLinearKHeads      int     `json:"linear_num_key_heads"`
	NLinearVHeads      int     `json:"linear_num_value_heads"`
	DLinearK           int     `json:"linear_key_head_dim"`
	DLinearV           int     `json:"linear_value_head_dim"`
	LinearConvKernel   int     `json:"linear_conv_kernel_dim"`
	PartialRotaryFactor float64 `json:"-"`
	MropeSection       []int   `json:"-"`
	ImageTokenID       int     `json:"image_token_id"`
}

// ropeParameters mirrors Qwen3.5's nested rope_parameters block.
type ropeParameters struct {
	RopeTheta           *float64 `json:"rope_theta"`
	PartialRotaryFactor *float64 `json:"partial_rotary_factor"`
	MropeSection        []int    `json:"mrope_section"`
}

type llmConfig struct {
	HiddenSize          int            `json:"hidden_size"`
	NumAttentionHeads   int            `json:"num_attention_heads"`
	NumKeyValueHeads    int            `json:"num_key_value_heads"`
	NumHiddenLayers     int            `json:"num_hidden_layers"`
	VocabSize           int            `json:"vocab_size"`
	HeadDim             int            `json:"head_dim"`
	RmsNormEps          float64        `json:"rms_norm_eps"`
	RopeTheta           float64        `json:"rope_theta"`
	RopeParameters      *ropeParameters `json:"rope_parameters"`
	IntermediateSize    *int           `json:"intermediate_size"`
	MoeIntermediateSize *int           `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize *int  `json:"shared_expert_intermediate_size"`
	NumExperts          int            `json:"num_experts"`
	NumExpertsPerTok    int            `json:"num_experts_per_tok"`
	LayerTypes          []string       `json:"layer_types"`
	LinearNumKeyHeads   int            `json:"linear_num_key_heads"`
	LinearNumValueHeads int            `json:"linear_num_value_heads"`
	LinearKeyHeadDim    int            `json:"linear_key_head_dim"`
	LinearValueHeadDim  int            `json:"linear_value_head_dim"`
	LinearConvKernelDim int            `json:"linear_conv_kernel_dim"`
	PartialRotaryFactor float64        `json:"partial_rotary_factor"`
	MropeSection        []int          `json:"mrope_section"`
}

// ReadConfig parses config.json from a model directory, handling both the
// flat layout and the nested text_config layout.
func ReadConfig(modelPath string) (*Config, error) {
	raw, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, err
	}
	var top struct {
		TextConfig        json.RawMessage `json:"text_config"`
		TieWordEmbeddings *bool           `json:"tie_word_embeddings"`
		ImageTokenID      *int            `json:"image_token_id"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}

	llmRaw := top.TextConfig
	if len(llmRaw) == 0 || string(llmRaw) == "null" {
		llmRaw = raw // flat layout
	}
	var lc llmConfig
	if err := json.Unmarshal(llmRaw, &lc); err != nil {
		return nil, fmt.Errorf("config.json text_config: %w", err)
	}

	cfg := &Config{
		NEmbed:           lc.HiddenSize,
		NHeads:           lc.NumAttentionHeads,
		NKVHeads:         lc.NumKeyValueHeads,
		NLayer:           lc.NumHiddenLayers,
		NVocab:           lc.VocabSize,
		DHead:            lc.HeadDim,
		NExperts:         lc.NumExperts,
		NExpertsPerToken: lc.NumExpertsPerTok,
		NMoeMlp:          intOr(lc.MoeIntermediateSize, 0),
		NSharedExpertMlp: intOr(lc.SharedExpertIntermediateSize, 0),
		LayerTypes:       lc.LayerTypes,
		NLinearKHeads:    lc.LinearNumKeyHeads,
		NLinearVHeads:    lc.LinearNumValueHeads,
		DLinearK:         lc.LinearKeyHeadDim,
		DLinearV:         lc.LinearValueHeadDim,
		LinearConvKernel: intOr2(lc.LinearConvKernelDim, 4),
		RmsNormEps:       lc.RmsNormEps,
		RopeTheta:        lc.RopeTheta,
		PartialRotaryFactor: lc.PartialRotaryFactor,
		MropeSection:     lc.MropeSection,
	}
	if top.TieWordEmbeddings != nil {
		cfg.TieWordEmbeddings = *top.TieWordEmbeddings
	}
	if top.ImageTokenID != nil {
		cfg.ImageTokenID = *top.ImageTokenID
	}
	if lc.RopeParameters != nil {
		if lc.RopeParameters.RopeTheta != nil {
			cfg.RopeTheta = *lc.RopeParameters.RopeTheta
		}
		if lc.RopeParameters.PartialRotaryFactor != nil {
			cfg.PartialRotaryFactor = *lc.RopeParameters.PartialRotaryFactor
		}
		if lc.RopeParameters.MropeSection != nil {
			cfg.MropeSection = lc.RopeParameters.MropeSection
		}
	}

	// dense MLP size: intermediate_size, else shared_expert size, else MoE size
	if lc.IntermediateSize != nil {
		cfg.NMlp = *lc.IntermediateSize
	} else if cfg.NSharedExpertMlp != 0 {
		cfg.NMlp = cfg.NSharedExpertMlp
	} else if cfg.NMoeMlp != 0 {
		cfg.NMlp = cfg.NMoeMlp
	}

	// defaults
	if cfg.DHead == 0 {
		cfg.DHead = cfg.NEmbed / cfg.NHeads
	}
	if cfg.PartialRotaryFactor == 0 {
		cfg.PartialRotaryFactor = 1.0
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.RmsNormEps == 0 {
		cfg.RmsNormEps = 1e-6
	}
	if cfg.MropeSection == nil {
		cfg.MropeSection = []int{11, 11, 10}
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (c *Config) validate() error {
	if c.NEmbed == 0 || c.NLayer == 0 || c.NVocab == 0 || c.NHeads == 0 {
		return fmt.Errorf("config: missing core fields (hidden_size/num_hidden_layers/vocab_size/num_attention_heads)")
	}
	if c.DHead <= 0 {
		return fmt.Errorf("config: head_dim missing")
	}
	if c.NKVHeads == 0 {
		c.NKVHeads = c.NHeads
	}
	if c.NKVHeads > c.NHeads || c.NHeads%c.NKVHeads != 0 {
		return fmt.Errorf("config: bad head counts %d/%d", c.NHeads, c.NKVHeads)
	}
	for i, lt := range c.LayerTypes {
		if lt != "full_attention" && lt != "linear_attention" {
			return fmt.Errorf("config: layer_types[%d]=%q unknown", i, lt)
		}
	}
	return nil
}

// LayerType returns the mixer type for a layer index.
func (c *Config) LayerType(i int) string {
	if i < len(c.LayerTypes) {
		return c.LayerTypes[i]
	}
	return "full_attention"
}

// RotaryDim is how many head dims get rotary treatment.
func (c *Config) RotaryDim() int {
	return int(float64(c.DHead) * c.PartialRotaryFactor)
}

func intOr(p *int, def int) int {
	if p == nil {
		return def
	}
	return *p
}

func intOr2(v, def int) int {
	if v == 0 {
		return def
	}
	return v
}

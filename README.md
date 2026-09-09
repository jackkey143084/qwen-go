# qwen-go

Pure-Go port of [Emericen/tiny-qwen](https://github.com/Emericen/tiny-qwen) — a minimal re-implementation of Qwen inference. Zero external dependencies. First-class FreeBSD target.

## What works now

- `safetensors` — HF safetensors reader: F32/F16/BF16/I8/U8 → float32, raw byte access for quantized tensors
- `tokenizer` — byte-level BPE from `tokenizer.json`, hand-rolled Qwen2 pre-tokenizer (Go regexp can't do the lookaheads), encode/decode/special tokens
- `model` — the full language stack:
  - RMSNorm + GemmaRMSNorm (`1+w`) + RMSNormGated
  - partial rotary embeddings (NeoX-style rotate_half), mRoPE sections collapse for text
  - GQA self-attention with per-head q/k norms and the Qwen3.5 q-gate output gating
  - GatedDeltaNet linear-attention mixer (causal depthwise conv, l2-norm, delta-rule recurrence with forget/surprise gating, fixed-size state cache)
  - dense SwiGLU MLP and MoE (softmax router, top-k renormalized, stacked 3D experts, gated shared expert)
  - QuantLinear int8/int4 (group-32, fp16 scales) — loading quantize.py-style dirs AND a Go quantizer (`quantizeInt8`/`quantizeInt4`)
  - greedy streaming generation, prefill + decode with KV/state caches
- `cmd/tiny-qwen` — streaming chat REPL with the Qwen chat template and `--think` toggle
- `cmd/qwen-openai` — OpenAI-compatible `/v1/chat/completions` passthrough server (streamed SSE and non-streamed) backed by the model, so any OpenAI client can talk to it:
  `GOGC=40 GOMEMLIMIT=2600MiB go run ./cmd/qwen-openai -model <dir> -addr :8080`

Tests: prefill/decode logit equivalence, GDN cache equivalence, quantization round-trip, MoE routing sanity.

## FreeBSD

Build natively on FreeBSD (Go 1.23+):

```
go build -o tiny-qwen ./cmd/tiny-qwen
./tiny-qwen -model /path/to/Qwen3-0.6B
```

FreeBSD-specific bits:

- weight shards are **mmap'd** (`model/mmap_unix.go`): the unified buffer cache pages weights in on demand, processes sharing a model share physical RAM, and `MADV_WILLNEED` (raw `SYS_MADVISE`, since Go's syscall package doesn't wrap it on FreeBSD) hints VM prefetch before prefill
- matmuls parallelize across goroutines (capped at 8 workers) — plays fine with `SCHED_ULE`
- cross-compile from anywhere: `GOOS=freebsd GOARCH=amd64 go build ./cmd/tiny-qwen` (arm64 too)

## Roadmap

See [ROADMAP.md](ROADMAP.md) / [ROADMAP_RU.md](ROADMAP_RU.md). Done through Phase 2 (text core + MoE + quantization). Remaining: vision tower, full agentic tool harness, OpenAI-compatible passthrough, perf passes.

## Not yet (honest list)

- vision encoder (phase 4) — text-only for now
- local agentic tool-calling harness (phase 5 — the run.py terminal loop; the `--url` passthrough side of phase 5 is done, this is the local agent loop that parses `<tool_call>` JSON and feeds results back)
- bf16-native kernels; weights upcast to float32 at load, so RAM ≈ 2× checkpoint size (mmap avoids the second file copy, not the upcast)

MIT licensed, like the original.

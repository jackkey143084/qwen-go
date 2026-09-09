# Roadmap: tiny-qwen → Go

Faithful port of [Emericen/tiny-qwen](https://github.com/Emericen/tiny-qwen) (Python/PyTorch) to pure Go, zero external dependencies. Ordered so every phase ends with something runnable.

## Phase 0 — Foundations ✅ (done, commit f601173)
- [x] `safetensors` — HF safetensors reader: F32/F16/BF16/I8/U8 → float32, raw byte access for quantized tensors
- [x] `tokenizer` — byte-level BPE from `tokenizer.json`, hand-rolled Qwen2 pre-tokenizer (Go regexp can't do the lookaheads), encode/decode/special tokens

## Phase 1 — Text core, dense model 🚧 (next)
Goal: greedy-stream a dense Qwen3-family model from a local checkpoint dir.
- [ ] `model/config.go` — parse `config.json` (flat + nested `text_config`)
- [ ] weight loading + renaming (`model.` → nothing, tie embeddings), state dict validation
- [ ] norms: RMSNorm + GemmaRMSNorm (`1+w` variant)
- [ ] partial rotary embeddings, GQA attention with per-head q/k norms + q-gate output gating
- [ ] KV cache (append + read at offset)
- [ ] dense SwiGLU MLP, tied lm head
- [ ] greedy streaming generation (argmax, same as Python's `temperature=0`)
- [ ] `cmd/tiny-qwen` — minimal chat REPL with the `im_start/im_end` template + thinking toggle
- [ ] validation: fixture logits from the Python impl on tiny random weights; smoke-test on Qwen3-0.6B

## Phase 2 — MoE + quantization
- [ ] MoE router (softmax over experts, top-k), batched expert MLPs (stacked 3D weights), shared expert
- [ ] `QuantLinear` int8/int4: group-wise dequant in the matmul hot path (scale file format: fp16 scales + uint8 packed weights)
- [ ] `--bits 4` quantize mode: read F32/BF16 checkpoint, emit quantized dir (Go version of `quantize_model`)
- [ ] validation: quantized Python checkpoint vs Go, token-by-token

## Phase 3 — GatedDeltaNet + mixed-layer models
Goal: run the full current `main` architecture generation (Qwen3.5–3.8).
- [ ] GDN layer: depthwise causal conv1d, l2 norm, gated delta rule recurrence, key/value gating, low-rank Q/K with a/b gates + dt bias
- [ ] per-layer dispatch from `layer_types` (`SelfAttention` vs `GatedDeltaNet`)
- [ ] state caching for GDN (recurrent state instead of KV)
- [ ] validation: single-layer parity harness vs Python for both mixer types

## Phase 4 — Vision
- [ ] image decode via Go stdlib (`image/jpeg`, `image/png` — no deps needed), bicubic resize, normalize
- [ ] dynamic-resolution resize (min/max pixel budget), 16px patchify with 2×2-merge ordering
- [ ] ViT tower: patch embedding, mRoPE (2D), full-attention blocks, pixel-merge → language tokens
- [ ] multimodal processor: `<|vision_start|>...<|vision_end|>` insertion, mRoPE 3-section position ids
- [ ] validation: the repo ships `test/data/qwen35_transformers_mm_outputs.json` fixtures — port those tests

## Phase 5 — Harness parity
- [ ] full agentic REPL from `run.py`: tool-call parsing/rendering, system prompts, chat history
- [ ] `--url/--api-key/--model` OpenAI-compatible passthrough mode (Go `net/http`)
- [ ] CLI flags matching the Python (`run.py Qwen/Qwen3.8-27B --bits 4 ...`)

## Phase 6 — Performance (only if it bothers someone)
- [ ] bf16-native matmul (no upcast), mmap-backed weights
- [ ] parallel matmul via goroutines, k-quants-friendly memory layout
- [ ] optional single-batch prefill speedups (compute-bound decode is fine for a tiny impl)

## Explicit non-goals (same as upstream)
Training, batching/serving many users, GPU acceleration. This is a readable reference implementation.

## Validation strategy
Every phase compares against the Python implementation: fixed inputs → logits/argmax tokens, using shipped test fixtures where available. A port that only "looks right" isn't done.

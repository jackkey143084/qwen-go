# qwen-go

Pure-Go port of [Emericen/tiny-qwen](https://github.com/Emericen/tiny-qwen) — a minimal re-implementation of Qwen inference. Zero external dependencies.

## Status (work in progress)

- [x] `safetensors` — Hugging Face safetensors reader (F32/F16/BF16/I8/U8, raw access for quantized tensors)
- [x] `tokenizer` — byte-level BPE from `tokenizer.json`, hand-rolled Qwen pre-tokenizer
- [ ] `model` — transformer stack (gated attention, GemmaRMSNorm, GatedDeltaNet, MoE, int4/int8 quantized linears)
- [ ] `cmd` — streaming chat harness

MIT licensed, like the original.

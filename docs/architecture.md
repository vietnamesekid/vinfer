# Architecture

```
request → server → scheduler → runtime → model → kernels
                       ↑           ↓
                  prefix cache ← state manager
```

| Component | Responsibility | Must not |
|-----------|----------------|----------|
| `server/` | OpenAI-compatible API, streaming | Contain batching logic |
| `scheduler/` | Queue, batching, prefill/decode budget | Know model architecture details |
| `runtime/` | Prefill/decode loop, sampler, stop conditions | Allocate cache memory directly |
| `state/` | Allocate, page, snapshot and free KV and recurrent state | Know about requests or SLOs |
| `models/` | Architecture definitions, weight loading | Hold state between calls |
| `kernels/` | Hot ops, each with a slow reference version | Change results beyond thresholds |
| `quant/` | Quantization, quantized weight loading | |

Every optimization is a flag in `src/vinfer/config.py`, off by default until it passes parity.

## Why this differs from a standard transformer engine

- **Two kinds of state.** The Qwen3.5/3.6 models are hybrid: Gated DeltaNet layers keep a
  fixed-size recurrent state per request, gated attention layers keep a KV cache per token.
  The state manager handles both from the start.
- **Prefix caching with recurrent state.** A KV cache can be cut at any position; a recurrent
  state can only be reused where a snapshot was taken.
- **MTP head.** Speculative decoding without a separate draft model. Recurrent state is
  updated in place, so a rejected draft token cannot simply be rolled back.
- **MLA (Kimi).** KV is cached as a compressed latent, a different layout from regular attention.
- **Sparse MoE.** Kimi routes each token to 8 of 384 experts; how tokens are grouped per
  expert decides performance, and the model needs expert sharding across GPUs.

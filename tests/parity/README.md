# Parity tests

Compare vinfer against HF Transformers on real weights. All tests here are `@pytest.mark.gpu`.

```bash
make parity MODEL=Qwen/Qwen3.5-2B   # GPU: HF against the same HF config, must be bit-exact
make noise  MODEL=Qwen/Qwen3.5-2B   # GPU: measure HF's own logit noise across configs
```

Both use the revision pinned in `bench/models.lock` unless `REVISION=<sha>` is given.

| File | Role |
|------|------|
| `prompts/v1.jsonl`, `prompts/long/` | 50 fixed prompts: short, Vietnamese + 8 other languages, code, multi-turn chat with system prompt, math, raw text, 4 long (frozen doc snapshots, >8k tokens). Never edit: add v2 |
| `logits.py` | `LogitSummary` (argmax/top-20/logsumexp everywhere, full logits at 32 sampled positions) and `compare` (max abs diff, KL, argmax mismatch) |
| `candidates.py` | `HFCandidate` with the noise knobs (dtype, batch, padding, order, attention impl) |
| `test_parity.py` | GPU parity test; reference summaries cached in `$VINFER_PARITY_CACHE` |
| `noise_floor.py` | Noise-floor measurement (`make noise`) |
| `thresholds.toml` | Parity thresholds |

The tooling itself is tested on CPU with a tiny random Qwen3.5 (`tests/unit/parity/`).

How thresholds are set:

1. Measure the noise floor: HF Transformers with different batch sizes, orderings, padding,
   attention implementations and dtypes.
2. Set logit thresholds per dtype from that noise floor.
3. Changes that do not alter the math (cache, batching, scheduling): argmax must match
   exactly on every position of the parity prompt set.
4. Changes that alter the math (quantization, lossy speculative decoding): judged by a quality
   eval, with a maximum allowed drop.

Until the per-dtype thresholds are set from a measured noise floor, only exact comparisons
are supported. Never loosen a threshold to make a change pass.

# Baselines

Launch scripts for the engines vinfer's speed is compared against. HF Transformers is the
correctness reference only (`make parity`), not a speed baseline.
Layout: `<engine>/configs/<config>.sh` (one launch config each, tuning flags only),
`<engine>/harness.sh` (settings the harness needs from every config), `<engine>/version.sh`,
`<engine>/requirements.txt`, and `common.sh` sourced by every config.
Never imported by `src/`.

| Engine | Version (pinned in `requirements.txt`) | Configs |
|--------|----------------------------------------|---------|
| vLLM | 0.33.0 | `default` (recipe as is), `tuned` (text-only + prefix cache), `tuned-mtp` |
| SGLang | 0.5.23 | `default` (cookbook as is), `tuned` (mamba radix extra_buffer, page 64), `tuned-mtp` |

Rules:

- Each engine has its own venv (`scripts/setup_baselines.sh`), because their torch pins conflict.
- Every flag lives in the launch scripts. The harness stores their text and sha256 in every
  result (`engine_config`), so a result always says exactly how the engine was run.
- `<engine>/harness.sh` turns on what the harness relies on: cached-token reporting
  (`--enable-prompt-tokens-details`, `--enable-cache-report`) for the prefix-cache hit rate,
  and vLLM's dev mode for `/reset_prefix_cache`.
- Each flag cites its source (recipe, cookbook, or the engine source at the pinned version).
  Flags are never written from memory.
- `tuned*` configs list the sweeps still to run on the benchmark machine. Record each sweep in
  `docs/benchmarking.md` before changing a flag.
- Output length is forced (`ignore_eos`), so every engine generates exactly the workload's
  `max_tokens`.

Inputs set by `bench/harness/engines.py`: `VINFER_MODEL`, `VINFER_PORT`, `VINFER_MODEL_REVISION`.
Run one by hand:

```bash
VINFER_MODEL=Qwen/Qwen3.5-2B VINFER_PORT=8000 bash bench/baselines/vllm/configs/tuned.sh
```

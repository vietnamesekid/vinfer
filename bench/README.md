# Benchmarks

Methodology and publishing rules: [docs/benchmarking.md](../docs/benchmarking.md).

| Dir | Contents |
|-----|----------|
| `harness/` | Load generator, metrics, server management (module map in `harness/__init__.py`) |
| `workloads/` | Fixed, seeded workloads W1–W4. Never edit one in place: add a new version |
| `baselines/` | Launch scripts for vLLM and SGLang with their exact configs |
| `results/` | One JSON per engine/workload/concurrency (committed), see [SCHEMA.md](results/SCHEMA.md). Per-request records go to `results/raw/` (ignored) |
| `HARDWARE.md` | The standard benchmark machine |

## Running

```bash
make workloads                      # build W1-W4 (CHECK=1 only verifies sha256 vs manifests)
make bench MODEL=Qwen/Qwen3.5-2B ENGINE=vllm CONFIG=tuned WORKLOAD="W1-v1 W4-v1"
make bench-all MODEL=Qwen/Qwen3.5-2B   # vllm/sglang x every config x W1-W4
uv run python -m bench.harness run --help    # every option (request rate, SLOs, num requests)
```

How a run works:

1. Start the server once from `baselines/<engine>/configs/<config>.sh` and wait for `/health`;
   every workload given with `--workload` is measured on that one launch.
2. Per concurrency level, warm up with at least one request per slot. On engines whose prefix
   cache can be reset (vLLM `/reset_prefix_cache` in dev mode, SGLang `/flush_cache`) the
   warmup reuses the workload's first requests, so kernels see the real shapes; elsewhere it
   uses synthetic requests that share no prefix with the workload.
3. Reset the prefix cache before **every** measured run, so no run starts with a warm cache.
   This reset is what makes reusing measured requests for warmup safe: never remove it.
4. Write one JSON per level. Results that fail validation (dirty tree, < 3 runs, not
   dedicated, failed or truncated requests, unknown version or revision) go to
   `results/raw/unofficial/`.

Timing definitions match `vllm bench serve` (see `results/SCHEMA.md`); `make validate` checks
that against each engine's own benchmark tool on a live server (`harness/validation.py`,
adapters in `harness/engine_benchmarks.py`).

Without a GPU: `uv run python -m bench.harness run --engine fake --workload W1-v1
--concurrency 1 8 --allow-dirty` measures the built-in fake server.

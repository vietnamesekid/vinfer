# Reproducing benchmarks

A number is only published if all of these hold:

1. Recorded per `bench/results/SCHEMA.md`: clean commit, model + HF revision, workload version,
   concurrency, provider, instance type, GPU, driver, CUDA, PyTorch, exact commands.
2. Measured on a **dedicated** instance (not spot or shared). Baselines re-run in the **same
   rental session** as the engine.
3. Baselines (vLLM, SGLang) are tuned properly, with versions and every flag recorded.
   A badly configured baseline makes the comparison meaningless.
4. At least 3 runs, with deviation reported. A single run never supports a "faster" claim.

If our numbers disagree with yours, please open a *Benchmark report* issue.

## Load model

Each workload is measured two ways:

- **Concurrency sweep** (throughput, TTFT, TPOT): concurrency 1, 8, 32, 128, closed loop
  (every request is ready at once, the number in flight is capped at the concurrency), as in
  the default mode of `vllm bench serve`. This is what `make bench-all` runs.
- **Arrival-rate sweep** (goodput): Poisson arrivals at several rates (`--request-rate`),
  scored against TTFT/TPOT service-level objectives. The objectives are not set yet; every run
  keeps per-request timings in `bench/results/raw/`, so goodput can be computed afterwards.

Requests per run default to `clamp(8 × concurrency, 64, 512)` (`--num-requests auto`), so
every concurrency slot is busy for several rounds without a level at concurrency 1 taking hours.

## Harness validation

Our numbers are only comparable with published ones if our harness measures like each
engine's own benchmark tool. `make validate` checks that:

```bash
make validate MODEL=Qwen/Qwen3.5-2B            # all engines; BENCHMARKS="vllm" for one
```

For each tool it launches that engine's baseline server, warms it up at each concurrency, then
alternates runs of the tool and of our harness (cache reset before each, and who goes first
alternates too) on the same single-turn prompts with the same fixed output length, and prints
mean ± stddev and the relative difference per metric.

| Tool | Engine | Adapter |
|------|--------|---------|
| `vllm bench serve` (v0.33.0) | vLLM | `VllmBenchServe` |
| `python -m sglang.benchmark.serving` (v0.5.23) | SGLang | `SglangBenchServing` |

Results: not published yet.

Adding an engine: subclass `EngineBenchmark` in `bench/harness/engine_benchmarks.py` (dataset
format, command, result parsing, with every flag checked in that engine's source at a pinned
version), add `bench/baselines/<engine>/` launch scripts and an entry in `ENGINES`
(`bench/harness/engines.py`), then register the benchmark in `ENGINE_BENCHMARKS`.

## Runbook: RunPod, 1× H100 80GB

Before renting: the repo is committed and pushed to a remote the machine can clone and push to.

1. Create a Secure Cloud on-demand pod with a network volume at `/workspace`; clone the repo;
   set `git config user.name/user.email` (needed to push results); export `HF_TOKEN`.
2. `VINFER_INSTANCE_TYPE="1x H100 80GB SXM" VINFER_DEDICATED=1 ./scripts/setup.sh`. It writes
   `.env.bench`, which `make` and the harness read automatically.
3. Correctness and tooling checks: `make parity MODEL=Qwen/Qwen3.5-2B`, `make noise
   MODEL=Qwen/Qwen3.5-2B` (once with and once without `flash-linear-attention` installed),
   `make validate MODEL=Qwen/Qwen3.5-2B`. Running these first surfaces setup problems within
   minutes, before the long run.
4. Baselines: `make bench-all MODEL=Qwen/Qwen3.5-2B` from a clean commit
   (`scripts/bench_matrix.sh` keeps going past a failed engine/config and lists failures).
5. `scripts/push_results.sh` before stopping the pod: it copies everything to the volume
   first, then pushes the official result files on a `results/...` branch.

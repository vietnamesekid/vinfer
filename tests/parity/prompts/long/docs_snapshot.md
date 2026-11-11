# vinfer

A from-scratch inference engine for open-weight **Qwen** and **Kimi** models, built to try and
measure each optimization technique on its own.

> **Status:** early. The benchmark harness, the correctness (parity) tooling and the baseline
> configurations are in place; the engine itself is not written yet.

## Goals, in priority order

1. **Correct.** Output matches the reference (HF Transformers) within thresholds derived from
   the measured noise floor.
2. **Measurable.** Every optimization has before/after numbers, reproducible with one command.
3. **Fast.** Approach or beat vLLM and SGLang on fixed workloads.

Out of scope: importing other serving engines into the engine (they are baselines only),
training and fine-tuning, production features (auth, multi-tenancy, autoscaling).

## Target models

| Model | Architecture |
|-------|--------------|
| Qwen3.5-2B / 9B, Qwen3.6-27B | Dense hybrid: Gated DeltaNet + gated attention, MTP head |
| Qwen3.6-35B-A3B | MoE hybrid, 256 experts (8 routed + 1 shared) |
| Kimi K2.x (INT4) | MoE, 1T total / 32B active, MLA, 384 experts |

## Benchmark results

None yet. Every published number comes with the commit, hardware, driver, the exact commands
and the full baseline configuration. See [docs/benchmarking.md](docs/benchmarking.md).

## Quickstart

```bash
git clone <repo-url> && cd vinfer
make install                              # every dependency group
make lint test                            # CPU only, no weights needed
make workloads                            # build W1-W4 from pinned datasets (~1 GB download)
uv run python -m bench.harness run --engine fake --workload W1-v1 --concurrency 1 8 --allow-dirty
```

The last command runs the benchmark harness against a built-in fake server, so the whole
measurement path can be tried without a GPU. On a GPU machine, see the runbook in
[docs/benchmarking.md](docs/benchmarking.md).

## Repository layout

| Path | Contents |
|------|----------|
| `src/vinfer/` | The engine: `server/`, `scheduler/`, `runtime/`, `state/`, `models/`, `kernels/`, `quant/`, and `config.py` with every optimization flag (off by default) |
| `bench/` | Benchmark harness, workloads W1–W4, baseline launch scripts, results |
| `tests/` | `unit/` (CPU), `parity/` (GPU + real weights), `regression/` |
| `experiments/` | One report per experiment, including failed ones |
| `docs/` | Architecture and benchmarking methodology |

## How this code is written

Most of the code is written by AI coding agents and checked by a person. It is not taken on
trust; it is checked by:

1. **Parity tests** against HF Transformers, with thresholds set from the measured noise floor.
2. **Reference kernels.** Every optimized kernel is tested against a slow, readable version.
3. **Experiment reports** for every optimization, plus human review of every PR.

## Licenses

- **Code:** [Apache-2.0](LICENSE).
- **Model weights are licensed separately and never committed here**: Qwen open-weight models
  are Apache-2.0, and Kimi K2.x is Modified MIT. Check each model's license before use.
- Code adapted from vLLM or SGLang (both Apache-2.0) keeps an attribution header in the file.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues as described in
[SECURITY.md](SECURITY.md).
# Contributing to vinfer

The rules below apply to humans and AI agents alike.

## Setup and checks

```bash
make install                  # uv sync with every dependency group the tests and tools need
make lint                     # ruff check + ruff format --check
make test                     # CPU unit + regression tests (what CI runs)
make parity MODEL=<hf-id>     # GPU + weights; run before AND after any change in src/
```

Tests needing a GPU or real weights are marked `@pytest.mark.gpu` and skipped without CUDA.

## Pull requests

- **One optimization per PR**, behind a flag in `src/vinfer/config.py` that is off by default
  until it passes parity, with a report in `experiments/EXP-NNN-*/`
  (see [TEMPLATE.md](experiments/TEMPLATE.md)).
- **Profile before optimizing**, and include the profile in the report.
- **Read the source before implementing.** Read the paper and the matching code of the engine
  you take the idea from. Never write kernel, CUDA or Triton code from memory.
- **Performance claims need numbers**: at least 3 runs against `main` on the same machine, with
  commit, hardware and commands ([docs/benchmarking.md](docs/benchmarking.md)).
- **Bug fixes** add a test in `tests/regression/` that fails without the fix. **New kernels**
  add a slow reference implementation and a test comparing the two.
- **Model configs:** before writing code for a checkpoint, read its `config.json` and
  modeling code.
- Never change tests, thresholds or workloads to make a change pass.
- Never import another serving engine (vLLM, SGLang, TGI, llama.cpp) into `src/`; they are
  baselines only (`ruff` enforces this).
- Code adapted from another project keeps its license header and a link to the source.

## Stop and ask a maintainer when

- A result looks too good (e.g. 2× faster from a small change).
- Parity fails and the cause is not found after two attempts.
- A change needs to move a component boundary ([docs/architecture.md](docs/architecture.md)).

## AI-written code

AI-generated contributions are welcome. You are still responsible for them:

- You must have read and understood every line, and be able to explain it in review.
- Say in the PR which parts were AI-generated and how you verified them.
- Don't submit benchmark numbers you did not run yourself.
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
# Result file schema

`bench/results/<YYYY-MM-DDTHHMMSS.mmm>_<engine>-<config>_<org--model>_<workload>_c<concurrency>.json`
(the start time of the run; an existing file is never overwritten)

Written by `bench/harness/benchmark.py`, validated by `bench/harness/result_file.py:validate`. A run that
fails validation (dirty tree, < 3 runs, not dedicated, failed requests, missing model revision)
goes to `bench/results/raw/unofficial/` with its `problems` listed, never to `bench/results/`.

```json
{
  "engine": "vllm | sglang | vinfer | external | fake",
  "engine_version": "git sha or package version",
  "engine_config": {"config": "default | tuned | tuned-mtp",
                    "launch_script": "bench/baselines/<engine>/configs/<config>.sh",
                    "launch_script_text": "exact config script, every tuning flag",
                    "harness_script_text": "bench/baselines/<engine>/harness.sh: flags the harness needs",
                    "launch_script_sha256": "sha256 of both scripts",
                    "extra_body": null, "cache_reset_between_runs": "/reset_prefix_cache | null"},
  "vinfer_commit": "full git sha (dirty tree not allowed)",
  "dirty": false,
  "model": "<hf-model-id>",
  "model_revision": "HF commit sha",
  "workload": "W1-v1",
  "concurrency": 32,
  "arrival": "closed-loop | poisson",
  "request_rate": "req/s for poisson, null for closed-loop",
  "seed": 0,
  "runs": 3,
  "num_requests": 256,
  "num_requests_rule": "auto | <int> | null (auto = clamp(8 x concurrency, 64, 512))",
  "warmup": 4,
  "command": "exact command line",
  "hardware": {
    "provider": "...", "instance_type": "...", "dedicated": true,
    "gpu": "...", "gpu_memory": "...", "num_gpus": 1, "driver": "...", "cuda": "...",
    "pod_id": "..."
  },
  "software": {"python": "...", "torch": "..."},
  "slo": {"ttft_ms": null, "tpot_ms": null},
  "metrics": {
    "ttft_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
    "tpot_ms": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
    "output_tokens_per_s": 0.0,
    "goodput_req_per_s": 0.0,
    "peak_vram_gb": 0.0,
    "prefix_cache_hit_rate": null,
    "itl_ms": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
    "e2el_ms": {"p50": 0.0, "p99": 0.0},
    "queue_ms": {"p50": 0.0, "p99": 0.0},
    "input_tokens_per_s": 0.0, "request_per_s": 0.0,
    "completed": 0, "failed": 0, "duration_s": 0.0
  },
  "stddev": {"...": "same keys as metrics: sample stddev across runs"},
  "per_run": [{"...": "metrics of each run"}],
  "raw_dir": "bench/results/raw/... (per-request records, not committed)",
  "problems": [],
  "timestamp": "ISO 8601"
}
```

Metric definitions:

Timing follows vLLM's `vllm bench serve` (v0.33.0) so the two can be compared (`make validate`):

- **TTFT**: request sent to the server → first output chunk. Time a request spends waiting for
  the client-side concurrency cap (or, in W4, for the previous turn) is `queue_ms`, not TTFT.
- **TPOT**: per request, (latency − TTFT) / (output tokens − 1); `tpot_ms.mean` averages requests.
- **Throughput**: output tokens/s for the whole system at the given concurrency.
- **Goodput**: requests/s meeting both the TTFT and TPOT SLOs.
- **Peak VRAM**: highest GPU memory during the run.
- **Prefix cache hit rate**: share of prompt tokens served from the prefix cache, from
  `usage.prompt_tokens_details.cached_tokens`; null if the server does not report it.
- **Peak VRAM**: NVML, all processes on the GPU. For vLLM/SGLang this mostly reflects memory
  reserved up front, read it with the engine config.

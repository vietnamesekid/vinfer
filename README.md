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
git clone https://github.com/vietnamesekid/vinfer.git && cd vinfer
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

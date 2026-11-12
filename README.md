# vinfer

A from-scratch inference engine for open-weight **Qwen** and **Kimi** models, built to try and
measure each optimization technique on its own.

> **Status:** early. A first, unoptimized engine (slow reference kernels in plain PyTorch)
> serves Qwen3 and the Qwen3.5 hybrid architecture and matches HF Transformers in fp32 on CPU.
> GPU parity in bf16, baseline benchmark numbers and every optimization are still to come.

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

## Progress

| Area | State |
|------|-------|
| Benchmark harness | Load generator, workloads W1–W4, vLLM/SGLang baseline configs; checked against `vllm bench serve` and SGLang's benchmark tool (one concurrency level still being reconciled) |
| Parity tooling | HF self-parity, layer-by-layer diff, bf16 thresholds set from the measured noise floor ([EXP-000](experiments/EXP-000-noise-floor/)) |
| Engine, correctness path | Qwen3 (standard attention) and Qwen3.5 (Gated DeltaNet + gated attention): loader, reference kernels, per-slot KV and recurrent state, batch-1 runtime, sampler |
| Kernel backends | Registry per op family, chosen by config and checked per device/dtype at startup; only the reference backends exist so far |
| Server | OpenAI-compatible `/v1/chat/completions` (streaming, usage, tool calls, reasoning split into `reasoning_content`) with per-model output parsers |
| GPU parity (bf16), baseline numbers, optimizations, larger and MoE models, Kimi | Not started |

### Correctness checks so far (CPU, fp32)

vinfer against HF Transformers on the same real weights, same prompt, fp32 on CPU (Apple
Silicon Mac), commit `3d30e6b`:

| Model (revision) | Prompt | Prefill logits, max abs diff | Argmax agreement | 32 greedy tokens |
|------------------|--------|------------------------------|------------------|------------------|
| Qwen3-0.6B (`c1899de2`) | 24 tokens | 1.3e-4 | 24/24 | identical |
| Qwen3.5-2B (`15852e8c`) | 24 tokens | 8.6e-5 | 24/24 | identical |
| Qwen3.5-2B (`15852e8c`) | 197 tokens (crosses a DeltaNet chunk) | 9.5e-5 | 197/197 | identical |

On Qwen3.5-2B the per-layer difference stays below 5.2e-6 relative across all 24 layers (18
DeltaNet, 6 attention). These are formula checks, not the bf16 GPU parity the thresholds are for,
and not speed measurements.

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

To try the engine itself on CPU (slow: reference kernels), with a downloaded checkpoint:

```bash
uv run python -m vinfer.server --model <checkpoint dir> --device cpu --dtype float32 --port 8000
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "any", "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false}
}'
```

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

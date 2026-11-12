# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Repository skeleton, CI, reference attention kernel.
- Benchmark harness (`bench/harness/`): streaming OpenAI client with timing matched to
  `vllm bench serve` v0.33.0, closed-loop and Poisson load, prefix-cache reset between runs,
  result validation, and a `validate` command that checks the harness against vLLM's and
  SGLang's own benchmark tools.
- Seeded workloads W1–W4 v1 built from pinned public datasets.
- Baseline launch scripts for vLLM 0.33.0 and SGLang 0.5.23.
- Parity tooling against HF Transformers, a fixed 50-prompt set, and a noise-floor script.
- RunPod setup and results push scripts.
- Engine, reference path (plain PyTorch, slow by design):
  - Qwen3 and Qwen3.5 (hybrid Gated DeltaNet + gated attention) models and checkpoint loader;
    the loader skips vision and MTP tensors, fails on missing or misshaped tensors, and keeps
    the tensors a bf16 checkpoint stores in fp32. Model notes in `docs/models/`.
  - Reference kernels with tests against HF Transformers: three RMSNorm variants, full and
    partial NeoX RoPE, SwiGLU, Gated DeltaNet (recurrent and chunked forms, short convolution).
  - Per-slot state pool holding the KV cache and the DeltaNet conv/SSM state.
  - Batch-1 runtime: prefill/decode loop, stop conditions, incremental detokenization, and a
    sampler (greedy, temperature, top-k, top-p) seeded per position.
  - `EngineConfig` as the single source of device and dtype; `build_engine`; the `CausalLM`
    interface the runtime, scheduler and parity tools depend on; `load_model` by model type.
  - Kernel backend registries (attention, Gated DeltaNet) selected by `KernelConfig` and
    checked against the device and dtype at startup; a contract test runs every registered
    backend against the reference in fp32 and bf16.
- OpenAI-compatible server (`python -m vinfer.server`): `/v1/chat/completions` with and without
  streaming, one chunk per generated token, usage (`include_usage`, `continuous_usage_stats`),
  OpenAI-shaped errors, `/v1/models`. Output parsers turn each model family's text into
  `content`, `reasoning_content` (also sent as `reasoning`) and `tool_calls`; Qwen3 and Qwen3.5
  parsers included, chosen automatically from the model type or with `--reasoning-parser` /
  `--tool-call-parser`. Tool calling works end to end. Unsupported request fields are rejected
  with 400 instead of being ignored.
- Serial scheduler: one request at a time on a worker thread; a disconnected client cancels its
  request.
- Parity: vinfer candidate (`make parity KERNELS=...`), comparison against HF fp32, and
  `tests.parity.layer_diff` to find the first diverging layer.
- Noise-floor report [EXP-000](experiments/EXP-000-noise-floor/) and bf16 parity thresholds,
  checked per prompt group (short prompts, long documents).
- Harness validation keeps per-request records from both sides.

### Fixed
- Harness validation compared different things: the prompts differed in set and order from
  each tool's, the httpx client added time to TTFT, and `vllm bench serve` sampled at
  temperature 1.0 while the harness ran greedy.
- The vLLM baseline venv lacked `vllm[bench]`.
- The parity environment did not record the causal_conv1d kernel.
- Parity aggregates could hide a NaN behind the worst-case max.

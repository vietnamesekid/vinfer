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

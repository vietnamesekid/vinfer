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

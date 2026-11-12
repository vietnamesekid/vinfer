# Experiments

One folder per experiment: `experiments/EXP-NNN-short-name/` containing a `README.md` copied from
[TEMPLATE.md](TEMPLATE.md), plus configs and profiles. Number experiments sequentially.
Failed experiments stay here too.

| ID | Title | Result | Headline |
|----|-------|--------|----------|
| [EXP-000](EXP-000-noise-floor/) | HF logit noise floor, Qwen3.5-2B | measured | bf16: batch size alone flips 7–13 argmax; eager vs SDPA 1.06%, bf16 vs fp32 1.47% |

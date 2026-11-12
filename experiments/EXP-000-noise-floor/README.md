# EXP-000: HF Transformers logit noise floor on Qwen3.5-2B

2026-11-11 · Commit `b054e93` · RunPod Secure Cloud, 1× H100 80GB HBM3 SXM (AP-IN-1) · `Qwen/Qwen3.5-2B`
@ `15852e8c` · Measured by agent, report written 2026-11-12

## Hypothesis

This is a measurement, not an optimization. The question: how far do HF Transformers logits move
when only things that should not change the result are changed (batch size, request order,
padding side, attention implementation, GDN kernels), and how far is bf16/fp16 from fp32?
Those numbers are the floor that the parity thresholds (`tests/parity/thresholds.toml`) are set from.

## Change

None in `src/`. Tool: `tests/parity/noise_floor.py` (`make noise`), unchanged since `b054e93`.

- Prompt set `tests/parity/prompts/v1`: 50 prompts, 48,528 positions in total. 46 short prompts
  (≤ 4,096 tokens, 1,308 positions) and 4 long documents (9,097–16,636 tokens, 47,220 positions).
- Each variant is compared with the plain config of the same dtype (bs=1, prompt order, right
  padding, SDPA). Two rows per dtype compare against fp32 instead: `<dtype>-vs-fp32`.
- Metrics (`tests/parity/logits.py`):
  - **argmax mismatch**: every position, teacher-forced on the prompt (not generated tokens).
  - **max/mean |Δlogit|** and **KL(ref‖cand)**: full logits at up to 32 positions per prompt
    (the last 16 plus 16 random), so about 1,600 positions in total.
  - Aggregate: worst prompt for max-type metrics, mean over prompts for mean-type metrics,
    totals for counts.
- GDN kernels cannot be switched in-process, so the tool ran once per installed combination:

| Run | GDN recurrence | Short conv | Raw file |
|-----|----------------|------------|----------|
| `torch` | PyTorch fallback | PyTorch fallback | `noise-floor-torch.json` |
| `fla` | flash-linear-attention 0.6.0 | PyTorch fallback | `noise-floor-fla.json` |
| `fla-cconv` | flash-linear-attention 0.6.0 | causal-conv1d 1.7.0 | `noise-floor-fla-cconv.json` |

Software in all three: torch 2.15.0+cu132, CUDA 13.2, transformers 5.22.2.

## Results

`repeat` (the plain config run twice) is bit-identical in every run and every dtype: HF is
deterministic at a fixed config, so all other differences below come from the knob changed.

### bf16

| Variant | Argmax mismatch (torch / fla / fla-cconv) | Rate, worst | max \|Δ\|, worst | KL mean, worst | KL max, worst |
|---|---|---|---|---|---|
| vs fp32 (precision gap) | 714 / 678 / 693 | 1.47% | 0.92 | 3.7e-4 | 3.0e-2 |
| batch 4 | 8 / 13 / 7 | 0.027% | 1.09 | 4.9e-4 | 4.0e-2 |
| batch 16 | **256** / 13 / 8 | 0.53% | 0.92 | 5.1e-4 | 4.0e-2 |
| batch 4, left padding | 162 / 188 / 194 | 0.40% | 1.19 | 5.4e-4 | 4.9e-2 |
| batch 4, shuffled order | 7 / 13 / 8 | 0.027% | 1.09 | 5.0e-4 | 4.0e-2 |
| eager attention | 493 / 511 / 512 | 1.06% | 1.19 | 5.0e-4 | 2.9e-2 |

### fp16

| Variant | Argmax mismatch (torch / fla / fla-cconv) | Rate, worst | max \|Δ\|, worst | KL mean, worst | KL max, worst |
|---|---|---|---|---|---|
| vs fp32 (precision gap) | 108 / 89 / 98 | 0.22% | 0.14 | 7.6e-6 | 7.7e-4 |
| batch 4 | 0 / 1 / 0 | 0.002% | 0.13 | 8.4e-6 | 7.0e-4 |
| batch 16 | 37 / 1 / 0 | 0.076% | 0.15 | 8.4e-6 | 7.0e-4 |
| batch 4, left padding | 20 / 22 / 29 | 0.060% | 0.13 | 8.9e-6 | 1.0e-3 |
| batch 4, shuffled order | 0 / 0 / 0 | 0 | 0.13 | 8.2e-6 | 7.0e-4 |
| eager attention | 67 / 74 / 56 | 0.15% | 0.16 | 7.7e-6 | 7.2e-4 |

### Where the mismatches are

Rate split by prompt length (bf16; short = 1,308 positions, long = 47,220):

| Variant | torch short / long | fla short / long | fla-cconv short / long |
|---|---|---|---|
| vs fp32 | 0.53% / 1.50% | 0.84% / 1.41% | 0.76% / 1.45% |
| batch 4 | 0.61% / 0 | 0.99% / 0 | 0.54% / 0 |
| batch 16 | 0.61% / 0.53% | 0.99% / 0 | 0.61% / 0 |
| batch 4, left padding | 0.61% / 0.33% | 0.92% / 0.37% | 0.69% / 0.39% |
| eager attention | 0.38% / 1.03% | 0.84% / 1.06% | 0.46% / 1.07% |

Observations:

1. **Batching alone flips argmax in bf16.** With batch 4 instead of 1, HF flips 7–13 positions,
   all in short prompts (6–10 of 46 short prompts, 1–2 flips each, often the same prompts across
   runs, e.g. `vi-4`). This happens in every GDN kernel combination, so it does not come from GDN.
2. **The long documents have many near-ties.** Most mismatches in every variant fall in the same
   four documents, and the first mismatch often lands on the same position across variants
   (e.g. `long-1` at 46, `long-2` at 10). Any perturbation that reaches those positions flips
   them, so the aggregate rate is dominated by four prompts.
3. **Left padding and eager attention change long-prompt results**, but batch 4 with right
   padding does not, with flash-linear-attention or the PyTorch fallback alike.
4. **PyTorch GDN fallback at batch 16 is the outlier**: 256 flips (154 in `long-0`, 94 in
   `long-1`) against 8–13 with flash-linear-attention. These runs do not show why.
5. **A different implementation is about as far as a lower precision.** Eager vs SDPA attention
   in bf16 (1.06%) is close to bf16 vs fp32 (1.40–1.47%). A vinfer forward pass with its own
   kernels is "a different implementation", so this row is the closest guide to what vinfer
   against HF will look like.
6. **fp16 is much tighter than bf16**: about 7× on max |Δlogit| and about 60× on mean KL. Qwen3.5
   ships bf16 weights; fp16 is reported for completeness.
7. **max |Δlogit| is about 0.8–1.2 in bf16 for every variant**, including the ones with almost no
   argmax flips. It separates "noise" from "bug" only coarsely; KL and argmax carry more
   information.

Raw results (git-ignored, kept off-repo with the session backup): `bench/results/raw/noise-floor-{torch,fla,fla-cconv}.json`.
`noise-floor-torch.pre-b054e93.json` is an earlier torch run without the `conv_backend` field;
its numbers are not used here. Commands, one per kernel combination, with the matching packages
installed:

```bash
make noise MODEL=Qwen/Qwen3.5-2B   # revision from bench/models.lock: 15852e8c...
```

## Conclusion

The noise floor is measured. What it means for parity:

- **HF itself does not keep argmax identical when only the batch size changes** (bf16: 7–13 flips
  from batch 1 to 4; fp16: 0–37). An "argmax must match exactly" rule for batching changes is
  only achievable if the engine is batch-invariant by construction.
- **A single aggregate rate hides the structure.** The 4 long documents hold 97% of the positions,
  so a bug confined to the short prompts could pass a whole-set rate. Thresholds are checked per
  prompt group (`short`, `long`).
- **HF fp32 is the closest to exact logits, but does not scale as the main reference**: a 27B model
  in fp32 needs about 108 GB.

Thresholds set from this report on 2026-11-12 (`tests/parity/thresholds.toml`, checked by
`tests/parity/thresholds.py`):

| bf16, against HF bf16 (bs=1, SDPA, flash-linear-attention + causal-conv1d) | `short` | `long` |
|---|---|---|
| Argmax mismatch rate | ≤ 2% | ≤ 2.2% |
| KL mean (mean over prompts) | ≤ 1.1e-3 | ≤ 1.1e-3 |
| KL max (worst prompt) | ≤ 0.1 | ≤ 0.1 |
| max \|Δlogit\| | reported only | reported only |

About 2× the worst same-dtype noise per group (argmax 0.99% short, 1.07% long; KL mean 5.8e-4
long, 5.6e-4 short; KL max 4.9e-2 short, 2.7e-2 long). The tables above aggregate over the whole
set; the per-group values come from the same raw files. bf16 vs fp32 is left out: against a
bf16 reference it is not noise (it reaches 1.50% on the long group). The ×2 leaves room for a second implementation's own
rounding while staying orders of magnitude below what a real bug does to KL. Changes that keep
the computation shape (same batch, same chunking) must stay bit-exact with the flag off; changes
that alter the shape (batching, prefix cache, chunked prefill) are held to the table above,
against the engine at batch 1. Batch invariance is not required. For models that fit in fp32
(2B, 9B), the distance from HF fp32 is also reported next to HF bf16's own distance.

Gaps in this measurement:

- No direct comparison between kernel combinations (e.g. `fla` bf16 vs `torch` bf16); each file
  compares only within itself.
- Argmax is measured teacher-forced on prompts. Decode positions after the prompt are not covered.
- GPU driver version is not recorded in the result environment.
- The ×2 margin is a judgment call, not derived from the data. If the engine's own kernels need
  more room, that is an experiment with its own report, never a quiet loosening.

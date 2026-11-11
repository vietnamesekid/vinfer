"""Measure the natural logit noise of HF Transformers, to set parity thresholds from.

Compares HF variants against an HF reference on the parity prompt set. Only things that should
not change the result are varied, so every difference is noise the engine will also face:

- batch size {1, 4, 16}, request order (shuffled), padding side (right/left),
- attention implementation (sdpa / eager),
- dtype: bf16 and fp16 runs against an fp32 reference (the "true" logits),
- GDN backend (fla kernels vs torch): not switchable in-process; run this script twice, once
  in an env with ``flash-linear-attention`` installed and once without, and compare the outputs.

Usage (GPU)::

    python -m tests.parity.noise_floor --model Qwen/Qwen3.5-2B --revision <sha>

Output (default ``bench/results/raw/noise-floor-<fla|torch>.json``): one JSON with, for each
variant, the aggregate and per-prompt metrics of ``tests/parity/logits.py:compare``, plus the
environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench.harness.config import pinned_revision
from tests.parity import prompt_set
from tests.parity.candidates import HFCandidate, HFConfig, environment, gdn_backend
from tests.parity.logits import LogitSummary, aggregate, compare


def variants(model: str, revision: str | None, dtype: str) -> dict[str, HFConfig]:
    """Configs that should give the same logits as the plain ``dtype`` baseline."""
    base = {"model": model, "revision": revision, "dtype": dtype}
    return {
        "repeat": HFConfig(**base),  # the baseline again: expected bit-identical
        "batch4": HFConfig(**base, batch_size=4),
        "batch16": HFConfig(**base, batch_size=16),
        "batch4-left-pad": HFConfig(**base, batch_size=4, padding_side="left"),
        "batch4-shuffled": HFConfig(**base, batch_size=4, order_seed=1),
        "eager-attention": HFConfig(**base, attn_implementation="eager"),
    }


def summaries(cfg: HFConfig, seqs: list[list[int]]) -> list[LogitSummary]:
    """Load the model for ``cfg``, summarise every prompt, free the GPU."""
    cand = HFCandidate(cfg)
    try:
        return cand.summaries(seqs)
    finally:
        del cand
        torch.cuda.empty_cache()


def metrics(keys: list[str], ref: list[LogitSummary], got: list[LogitSummary]) -> dict:
    per = {k: compare(r, g) for k, r, g in zip(keys, ref, got, strict=True)}
    return {"aggregate": aggregate(per), "per_prompt": per}


def run(model: str, revision: str | None, out_path: Path, dtypes: list[str]) -> dict:
    """Measure every variant; the JSON at ``out_path`` is rewritten after each one."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    ids = prompt_set.tokenize(prompt_set.load("v1"), tok)
    keys, seqs = list(ids), list(ids.values())
    result: dict = {
        "model": model,
        "revision": revision,
        "environment": environment(),
        "prompt_tokens": {k: len(v) for k, v in ids.items()},
        "variants": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def record(label: str, ref: list[LogitSummary], got: list[LogitSummary]) -> None:
        result["variants"][label] = metrics(keys, ref, got)
        print(label, json.dumps(result["variants"][label]["aggregate"]), flush=True)
        out_path.write_text(json.dumps(result, indent=2))

    fp32 = summaries(HFConfig(model, revision, "float32"), seqs)  # closest to exact logits
    for dtype in dtypes:
        base = summaries(HFConfig(model, revision, dtype), seqs)
        record(f"{dtype}-vs-fp32", fp32, base)  # precision gap
        for name, cfg in variants(model, revision, dtype).items():
            record(f"{dtype}:{name}", base, summaries(cfg, seqs))  # same-dtype noise
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--revision", help="default: the pin in bench/models.lock")
    p.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"])
    # Default under bench/results/raw/ (git-ignored): writing into experiments/ mid-session
    # would make the tree dirty and block official benchmark runs until a commit. The GDN
    # backend is in the name, so the runs with and without flash-linear-attention both stay.
    p.add_argument("--out", type=Path, help="default: bench/results/raw/noise-floor-<gdn>.json")
    a = p.parse_args()
    # Same weights as parity and the baselines: never silently fall back to `main`.
    revision = a.revision or pinned_revision(a.model)
    if revision is None:
        raise SystemExit(f"no revision for {a.model}: pin it in bench/models.lock or pass one")
    out = a.out or Path(f"bench/results/raw/noise-floor-{gdn_backend()}.json")
    run(a.model, revision, out, a.dtypes)


if __name__ == "__main__":
    main()

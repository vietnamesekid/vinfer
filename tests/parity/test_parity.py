"""Parity on real weights (GPU).

HF against the same HF configuration must be bit-identical; this checks the tooling itself.
The vinfer engine is compared here too once it exists, with ``thresholds.check``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.parity import prompt_set, thresholds
from tests.parity.candidates import HFCandidate, HFConfig, environment
from tests.parity.logits import LogitSummary, aggregate, compare

THRESHOLDS = thresholds.load()


def reference_dir(env: dict, prompt_version: str = "v1") -> Path:
    """Cache directory for HF reference summaries.

    The key covers model, revision, dtype, prompt set and the software/GPU environment, so a
    reference computed with another GPU or library version is never reused for a bit-exact test.
    """
    key = {
        "model": env["model"],
        "revision": env["revision"],
        "dtype": env["dtype"],
        "prompts": prompt_version,
        **environment(),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return env["cache"] / f"{env['model'].replace('/', '--')}_{env['dtype']}_{digest}"


def reference_summaries(
    env: dict, ids: dict[str, list[int]], model: HFCandidate
) -> dict[str, LogitSummary]:
    """Cached HF reference summaries; computed with ``model`` (already loaded) on a miss."""
    d = reference_dir(env)
    if all((d / f"{k}.safetensors").exists() for k in ids):
        return {k: LogitSummary.load(d / f"{k}.safetensors") for k in ids}
    sums = dict(zip(ids, model.summaries(list(ids.values())), strict=True))
    for k, s in sums.items():
        s.save(d / f"{k}.safetensors")
    (d / "meta.json").write_text(json.dumps({"name": model.name, **environment()}, indent=2))
    return sums


@pytest.mark.gpu
def test_hf_self_parity(parity_env):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(parity_env["model"], revision=parity_env["revision"])
    prompts = prompt_set.load("v1")
    ids = prompt_set.tokenize(prompts, tok)
    # One model on the GPU: it computes the reference on a cache miss, then the candidate.
    hf = HFCandidate(HFConfig(parity_env["model"], parity_env["revision"], parity_env["dtype"]))
    ref = reference_summaries(parity_env, ids, hf)
    got = dict(zip(ids, hf.summaries(list(ids.values())), strict=True))
    per = {k: compare(ref[k], got[k]) for k in ids}
    agg = aggregate(per)
    groups = thresholds.group_aggregates(per, thresholds.prompt_groups(prompts))
    print(json.dumps({"all": agg, **groups}, indent=2))
    # Same config, same machine: anything but bit-identical means the tooling is broken
    # (or HF is non-deterministic, which the noise-floor report must then say).
    assert agg["argmax_mismatch"] == THRESHOLDS["exact_math"]["argmax_mismatch"]
    assert agg["bitwise_equal"], {k: v for k, v in per.items() if not v["bitwise_equal"]}

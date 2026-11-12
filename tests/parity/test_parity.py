"""Parity on real weights (GPU).

- HF against the same HF configuration must be bit-identical; this checks the tooling itself.
- vinfer against HF in the same dtype must pass ``thresholds.check`` (per prompt group).
- vinfer's distance from HF fp32 (the closest to exact logits) must stay within twice HF's own
  distance in the same dtype. Beyond that, stop and ask: vinfer would be less accurate than a
  second bf16 implementation should be.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.parity import prompt_set, thresholds
from tests.parity.candidates import Candidate, HFCandidate, HFConfig, VinferCandidate, environment
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
    env: dict, ids: dict[str, list[int]], model: Candidate | Callable[[], Candidate]
) -> dict[str, LogitSummary]:
    """Cached HF reference summaries. On a miss they are computed with ``model``: a loaded
    candidate, or a function that loads one (then freed again, to leave the GPU to the next)."""
    d = reference_dir(env)
    if all((d / f"{k}.safetensors").exists() for k in ids):
        return {k: LogitSummary.load(d / f"{k}.safetensors") for k in ids}
    loaded = model() if callable(model) else model
    sums = dict(zip(ids, loaded.summaries(list(ids.values())), strict=True))
    for k, s in sums.items():
        s.save(d / f"{k}.safetensors")
    (d / "meta.json").write_text(json.dumps({"name": loaded.name, **environment()}, indent=2))
    if callable(model):
        del loaded
        _free_gpu()
    return sums


def _free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prompts(env: dict) -> tuple[list[dict], dict[str, list[int]]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(env["model"], revision=env["revision"])
    prompts = prompt_set.load("v1")
    return prompts, prompt_set.tokenize(prompts, tok)


def _hf_reference(env: dict, ids: dict[str, list[int]], dtype: str) -> dict[str, LogitSummary]:
    env = {**env, "dtype": dtype}
    return reference_summaries(
        env, ids, lambda: HFCandidate(HFConfig(env["model"], env["revision"], dtype))
    )


def _report(label: str, per: dict, prompts: list[dict]) -> dict:
    groups = thresholds.group_aggregates(per, thresholds.prompt_groups(prompts))
    print(label, json.dumps({"all": aggregate(per), **groups}, indent=2))
    return groups


@pytest.mark.gpu
def test_hf_self_parity(parity_env):
    prompts, ids = _prompts(parity_env)
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


@pytest.fixture(scope="module")
def vinfer_run(parity_env):
    """vinfer summaries on the parity set, computed once for the tests below."""
    from huggingface_hub import snapshot_download

    prompts, ids = _prompts(parity_env)
    path = snapshot_download(
        parity_env["model"],
        revision=parity_env["revision"],
        allow_patterns=["*.json", "*.safetensors"],
    )
    vinfer = VinferCandidate(path, parity_env["dtype"], kernels=parity_env["kernels"])
    print("candidate:", vinfer.name)
    sums = dict(zip(ids, vinfer.summaries(list(ids.values())), strict=True))
    del vinfer
    _free_gpu()
    return prompts, ids, sums


@pytest.mark.gpu
def test_vinfer_parity(parity_env, vinfer_run):
    prompts, ids, got = vinfer_run
    limits = THRESHOLDS.get(parity_env["dtype"])
    if limits is None:
        pytest.fail(f"no parity thresholds for {parity_env['dtype']} in thresholds.toml")
    ref = _hf_reference(parity_env, ids, parity_env["dtype"])
    per = {k: compare(ref[k], got[k]) for k in ids}
    _report("vinfer vs HF", per, prompts)
    failures = thresholds.check(per, thresholds.prompt_groups(prompts), limits)
    assert not failures, failures


@pytest.mark.gpu
def test_vinfer_vs_fp32_within_twice_hf(parity_env, vinfer_run):
    prompts, ids, got = vinfer_run
    exact = _hf_reference(parity_env, ids, "float32")
    hf = _hf_reference(parity_env, ids, parity_env["dtype"])
    ours = _report("vinfer vs HF fp32", {k: compare(exact[k], got[k]) for k in ids}, prompts)
    theirs = _report("HF vs HF fp32", {k: compare(exact[k], hf[k]) for k in ids}, prompts)
    too_far = [
        f"{group}: {metric} {ours[group][metric]:.3g} > 2 x {theirs[group][metric]:.3g}"
        for group in ours
        for metric in ("argmax_mismatch_rate", "kl_mean")
        if not ours[group][metric] <= 2 * theirs[group][metric]
    ]
    assert not too_far, too_far

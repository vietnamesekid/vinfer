"""The parity thresholds file and the per-group check (``tests/parity/thresholds.py``)."""

from __future__ import annotations

import math

import pytest

from tests.parity import prompt_set, thresholds
from tests.parity.logits import aggregate

LIMITS = thresholds.load()["bfloat16"]

# Position counts of the v1 set as tokenized for Qwen3.5-2B (EXP-000): 46 short prompts with
# 1,308 positions in all, 4 long documents with 47,220.
SHORT_POSITIONS = [28] * 45 + [48]
LONG_POSITIONS = [11805] * 4


def _metrics(positions: int, mismatch: int = 0, kl_mean: float = 1e-4, kl_max: float = 1e-2):
    return {
        "positions": positions,
        "argmax_mismatch": mismatch,
        "argmax_mismatch_rate": mismatch / positions,
        "first_mismatch_pos": 0 if mismatch else -1,
        "max_abs_logit": 1.0,
        "mean_abs_logit": 0.03,
        "kl_mean": kl_mean,
        "kl_max": kl_max,
        "lse_max_abs": 0.1,
        "bitwise_equal": False,
    }


def _v1(short_mismatch=(), long_mismatch=(), **kw):
    """Per-prompt metrics shaped like the v1 set; ``*_mismatch`` gives flips per prompt."""
    per, groups = {}, {}
    for group, sizes, flips in (
        ("short", SHORT_POSITIONS, short_mismatch),
        ("long", LONG_POSITIONS, long_mismatch),
    ):
        flips = list(flips) + [0] * (len(sizes) - len(flips))
        for i, (n, m) in enumerate(zip(sizes, flips, strict=True)):
            per[f"{group}{i}"] = _metrics(n, m, **kw)
            groups[f"{group}{i}"] = group
    return per, groups


def test_fixture_matches_the_v1_set():
    assert sum(SHORT_POSITIONS) == 1308 and sum(LONG_POSITIONS) == 47220


def test_thresholds_are_decided_for_both_groups_of_the_v1_set():
    cfg = thresholds.load()
    assert cfg["status"] == "decided"
    groups = thresholds.prompt_groups(prompt_set.load("v1"))
    assert sorted(groups.values()).count("long") == 4 and len(groups) == 50
    assert set(cfg["bfloat16"]["argmax_mismatch_rate"]) == set(groups.values())


def test_worst_measured_noise_passes():
    # EXP-000 worst same-dtype bf16 noise, per group: short 13/1,308 flips (0.99%), long
    # 505/47,220 (1.07%), KL mean 5.8e-4, KL max 4.9e-2.
    per, groups = _v1(
        short_mismatch=[1] * 13, long_mismatch=[127, 126, 126, 126], kl_mean=5.8e-4, kl_max=4.9e-2
    )
    aggs = thresholds.group_aggregates(per, groups)
    assert aggs["short"]["argmax_mismatch"] == 13 and aggs["long"]["argmax_mismatch"] == 505
    assert thresholds.check(per, groups, LIMITS) == []


def test_bug_in_short_prompts_fails_although_the_overall_rate_is_low():
    # Half the short prompts fully wrong: about 1.3% of all positions, under a whole-set limit
    # near the noise, but 50% of the short group.
    per, groups = _v1(short_mismatch=[28] * 23)
    assert aggregate(per)["argmax_mismatch_rate"] < LIMITS["argmax_mismatch_rate"]["long"]
    failures = thresholds.check(per, groups, LIMITS)
    assert failures and all(f.startswith("short: argmax") for f in failures)


def test_large_kl_fails():
    per, groups = _v1(kl_max=0.5)
    assert any("kl_max" in f for f in thresholds.check(per, groups, LIMITS))


@pytest.mark.parametrize("metric", ["kl_mean", "kl_max"])
def test_nan_fails_wherever_it_is(metric):
    # A NaN after a finite value: the builtin max() would return the finite one.
    per, groups = _v1()
    per["long3"][metric] = math.nan
    assert any(f"long: {metric}" in f for f in thresholds.check(per, groups, LIMITS))


def test_a_partial_run_is_an_error_not_a_pass():
    per, groups = _v1()
    one_each = {k: v for k, v in per.items() if k in ("short0", "long0")}
    with pytest.raises(ValueError, match="not measured"):
        thresholds.check(one_each, groups, LIMITS)


def test_unmapped_prompt_or_unknown_group_is_reported():
    per, groups = _v1()
    with pytest.raises(ValueError, match="no group"):
        thresholds.check({**per, "extra": _metrics(10)}, groups, LIMITS)
    per["new0"], groups["new0"] = _metrics(10), "tool-call"
    with pytest.raises(ValueError, match="no argmax_mismatch_rate limit"):
        thresholds.check(per, groups, LIMITS)

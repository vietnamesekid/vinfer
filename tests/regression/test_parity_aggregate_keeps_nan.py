"""``logits.aggregate`` must not hide a NaN behind the worst-case max.

Bug: it used the builtin ``max()``, which compares with ``>``. Every comparison with NaN is false,
so ``max([0.01, nan])`` returned 0.01: a prompt whose logits went NaN vanished from ``kl_max`` and
``max_abs_logit`` whenever a finite prompt came first.
"""

from __future__ import annotations

import math

import pytest

from tests.parity.logits import aggregate


def _metrics(**override):
    base = {
        "positions": 10,
        "argmax_mismatch": 0,
        "max_abs_logit": 0.5,
        "mean_abs_logit": 0.01,
        "kl_mean": 1e-4,
        "kl_max": 1e-2,
        "bitwise_equal": False,
    }
    return base | override


@pytest.mark.parametrize("metric", ["kl_max", "max_abs_logit"])
def test_nan_after_a_finite_prompt_stays_nan(metric):
    agg = aggregate({"a": _metrics(), "b": _metrics(**{metric: math.nan})})
    assert math.isnan(agg[metric])

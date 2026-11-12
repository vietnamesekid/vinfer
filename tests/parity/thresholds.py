"""Parity thresholds (``thresholds.toml``) and the check that applies them.

Prompts are judged in two groups. The 4 long documents hold 97% of the positions of the v1
set, so one rate over the whole set would let a bug that breaks every short prompt (2.7% of
the positions) pass. The values come from the noise floor in ``experiments/EXP-000-noise-floor/``.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tests.parity.logits import aggregate

PATH = Path(__file__).parent / "thresholds.toml"


def load(path: Path = PATH) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def prompt_groups(prompts: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Prompt id -> threshold group: ``long`` for the long documents, ``short`` for the rest."""
    return {p["id"]: "long" if p["kind"] == "long" else "short" for p in prompts}


def group_aggregates(
    per_prompt: dict[str, dict[str, Any]], groups: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """``logits.aggregate`` over each group of prompts."""
    if unmapped := set(per_prompt) - set(groups):
        raise ValueError(f"prompt(s) with no group: {sorted(unmapped)}")
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for key, metrics in per_prompt.items():
        by_group.setdefault(groups[key], {})[key] = metrics
    return {g: aggregate(p) for g, p in sorted(by_group.items())}


def check(
    per_prompt: dict[str, dict[str, Any]], groups: dict[str, str], limits: dict[str, Any]
) -> list[str]:
    """Violations of ``limits`` (one dtype's table of ``thresholds.toml``); empty means pass.

    ``groups`` maps every prompt of the set, and ``per_prompt`` must cover all of them: judging a
    partial set (a crashed run, a filtered dict) would pass on whatever is left. Raises
    ``ValueError`` for that and for a group with no limit. ``max_abs_logit`` is reported only
    (see ``thresholds.toml``).
    """
    if missing := set(groups) - set(per_prompt):
        raise ValueError(f"{len(missing)} prompt(s) not measured, e.g. {sorted(missing)[:5]}")
    aggs = group_aggregates(per_prompt, groups)
    rate_limits = limits["argmax_mismatch_rate"]
    if no_limit := set(aggs) - set(rate_limits):
        raise ValueError(f"no argmax_mismatch_rate limit for group(s) {sorted(no_limit)}")
    if no_prompts := set(rate_limits) - set(aggs):
        raise ValueError(f"no prompts in group(s) {sorted(no_prompts)}")
    failures = []
    for group, agg in aggs.items():
        checks = {
            "argmax_mismatch_rate": rate_limits[group],
            "kl_mean": limits["kl_mean"],
            "kl_max": limits["kl_max"],
        }
        for metric, limit in checks.items():
            # `not <=` so that NaN fails (aggregate propagates NaN): a NaN is a bug, never noise.
            if not agg[metric] <= limit:
                failures.append(f"{group}: {metric} {agg[metric]:.3g} > {limit:.3g}")
    return failures

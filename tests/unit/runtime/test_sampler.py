"""Sampler: greedy, the sampled distribution, seeds, top-k and top-p (CPU)."""

from __future__ import annotations

import pytest
import torch

from vinfer.runtime.sampler import SamplingParams, filtered_probs, position_seed, sample

LOGITS = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0, -3.0])


def _frequencies(params: SamplingParams, n: int = 20000, seed: int = 1) -> torch.Tensor:
    counts = torch.zeros(len(LOGITS))
    for pos in range(n):
        counts[sample(LOGITS, params, seed, pos)] += 1
    return counts / n


def test_greedy_is_argmax():
    assert sample(torch.tensor([0.1, 3.0, -2.0]), SamplingParams(temperature=0), 0, 0) == 1


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=1.0),
        SamplingParams(temperature=0.5),
        SamplingParams(temperature=1.0, top_k=3),
        SamplingParams(temperature=1.0, top_p=0.8),
    ],
)
def test_sampled_frequencies_match_the_filtered_distribution(params):
    # Deterministic (fixed seed), so no flakiness; 20,000 draws put the total variation of an
    # unbiased sampler well under 0.015.
    expected = filtered_probs(LOGITS, params)
    tv = 0.5 * (_frequencies(params) - expected).abs().sum()
    assert tv < 0.015, (tv, expected)


def test_top_k_and_top_p_remove_tokens():
    assert int((filtered_probs(LOGITS, SamplingParams(top_k=2)) > 0).sum()) == 2
    probs = torch.softmax(LOGITS, -1)
    kept = filtered_probs(LOGITS, SamplingParams(top_p=0.8)) > 0
    # Smallest prefix of the sorted tokens with mass >= 0.8.
    n = int((probs.sort(descending=True).values.cumsum(0) < 0.8).sum()) + 1
    assert int(kept.sum()) == n and bool(kept[:n].all())
    assert int((filtered_probs(LOGITS, SamplingParams(top_p=1e-6)) > 0).sum()) == 1


def test_same_seed_same_token_and_seed_depends_on_position():
    params = SamplingParams(temperature=1.0)
    assert [sample(LOGITS, params, 7, p) for p in range(50)] == [
        sample(LOGITS, params, 7, p) for p in range(50)
    ]
    assert position_seed(7, 0) != position_seed(7, 1) != position_seed(8, 0)


def test_invalid_params_are_rejected():
    for kw in ({"temperature": -1}, {"top_p": 0}, {"top_k": -1}, {"max_tokens": 0}):
        with pytest.raises(ValueError):
            SamplingParams(**kw)

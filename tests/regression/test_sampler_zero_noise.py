"""The sampler must never pick a token that top-k/top-p removed, even if a noise draw is 0.

Bug: ``argmax(p / E)`` with ``E = 0`` turned a filtered token (``p = 0``) into ``0/0 = NaN``,
and ``torch.argmax`` treats NaN as the maximum.
"""

from __future__ import annotations

import torch

from vinfer.runtime.sampler import SamplingParams, sample


def test_zero_noise_does_not_select_a_filtered_token(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "exponential_", lambda self, generator=None: self.zero_())
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    for pos in range(5):
        assert sample(logits, SamplingParams(top_k=2), seed=0, position=pos) in (0, 1)
        assert sample(logits, SamplingParams(top_p=0.5), seed=0, position=pos) == 0

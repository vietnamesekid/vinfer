"""Token sampling: greedy, temperature, top-k, top-p, with per-position seeds.

Random sampling uses the exponential race: ``argmax(p_i / E_i)`` with ``E_i ~ Exp(1)`` i.i.d.
picks token ``i`` with probability ``p_i`` (the same distribution as ``multinomial``), without a
host-side draw. The noise ``E`` comes from a generator seeded by ``(seed, position)``, so a
request's output depends only on its seed and its own tokens, not on what else is in the batch
or on how many random numbers other requests consumed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0  # 0 = greedy
    top_k: int = 0  # 0 = off
    top_p: float = 1.0  # 1 = off
    seed: int | None = None  # None: the runtime picks one per request
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


def position_seed(seed: int, position: int) -> int:
    """A 63-bit seed derived from ``(seed, position)`` (stable across runs and platforms)."""
    digest = hashlib.blake2b(f"{seed}:{position}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def filtered_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Probabilities after temperature, top-k and top-p, fp32, over the last dim."""
    logits = logits.float() / params.temperature
    if params.top_k and params.top_k < logits.shape[-1]:
        kth = torch.topk(logits, params.top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if params.top_p < 1:
        # Keep the smallest set of most likely tokens whose mass reaches top_p: drop a token when
        # the mass strictly before it already does. The most likely token is always kept.
        sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
        before = sorted_probs.cumsum(-1) - sorted_probs
        sorted_probs = sorted_probs.masked_fill(before >= params.top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


def sample(logits: torch.Tensor, params: SamplingParams, seed: int, position: int) -> int:
    """One token id from ``logits`` (``[vocab]``) for the token at ``position``."""
    if params.greedy:
        return int(torch.argmax(logits))
    probs = filtered_probs(logits, params)
    gen = torch.Generator(device=probs.device).manual_seed(position_seed(seed, position))
    noise = torch.empty_like(probs).exponential_(generator=gen)
    return int(torch.argmax(probs / noise))

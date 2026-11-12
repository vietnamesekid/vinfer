"""The attention backend interface and its registry."""

from __future__ import annotations

from typing import Protocol

import torch

from vinfer.kernels.registry import BackendRegistry


class AttentionBackend(Protocol):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        start: int,
    ) -> torch.Tensor:
        """Write ``k``, ``v`` at positions ``start .. start+T-1`` of ``kv_cache``, then return
        causal attention of ``q`` over positions ``[0, start+T)``.

        Shapes: ``q`` ``[Hq, T, D]``; ``k``, ``v`` ``[Hkv, T, D]``; each cache tensor
        ``[Hkv, capacity, D]`` (a view into the state pool). Returns ``[Hq, T, D]`` in ``q``'s
        dtype. The cache write belongs to the backend because fast kernels fuse it with the
        attention (e.g. ``flash_attn_with_kvcache``) and paged layouts decide where it goes.
        """
        ...


ATTENTION: BackendRegistry[AttentionBackend] = BackendRegistry("attention")

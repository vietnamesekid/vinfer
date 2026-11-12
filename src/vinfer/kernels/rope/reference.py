"""Slow, readable rotary position embedding (NeoX layout), full or partial.

NeoX layout: the rotated dims are split into two halves ``(x1, x2)`` and rotated as
``(x1 cos - x2 sin, x2 cos + x1 sin)``, with ``cos``/``sin`` repeated over both halves. Partial
RoPE (Qwen3.5: 64 of 256 dims) rotates the first ``rotary_dim`` dims and passes the rest through.

Frequencies and angles are computed in fp32; ``cos``/``sin`` are cast to the model dtype before
they are applied, as HF Transformers does.
"""

from __future__ import annotations

import torch


def rope_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    """``1 / theta^(2i / rotary_dim)`` for ``i < rotary_dim / 2``, fp32."""
    if rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))


def rope_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos``, ``sin`` of shape ``[*positions.shape, rotary_dim]`` in ``dtype``."""
    freqs = positions.float()[..., None] * inv_freq.to(positions.device)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``cos.shape[-1]`` dims of ``x``; the remaining dims pass through.

    ``cos``/``sin`` must broadcast against ``x[..., :rotary_dim]`` (e.g. ``[seq, rotary_dim]``
    against ``[heads, seq, head_dim]``).
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot[..., : rotary_dim // 2], x_rot[..., rotary_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)

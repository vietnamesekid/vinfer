"""Slow, readable reference attention. Optimized kernels are tested against this."""

from __future__ import annotations

import math

import torch


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Naive multi-head attention with grouped-query support.

    Args:
        q: ``[batch, num_q_heads, q_len, head_dim]``
        k, v: ``[batch, num_kv_heads, kv_len, head_dim]``. ``num_q_heads`` must be a
            multiple of ``num_kv_heads`` (GQA).
        causal: Apply a causal mask aligned to the end of the KV sequence, so a decode
            step (``q_len == 1``) attends to every cached token.
        scale: Softmax scale, defaults to ``1 / sqrt(head_dim)``.

    Returns:
        ``[batch, num_q_heads, q_len, head_dim]`` in the dtype of ``q``.
    """
    num_q_heads, q_len, head_dim = q.shape[1], q.shape[2], q.shape[3]
    num_kv_heads, kv_len = k.shape[1], k.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads={num_q_heads} not divisible by num_kv_heads={num_kv_heads}")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    group = num_q_heads // num_kv_heads
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)

    # Compute in fp32 for a stable reference.
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        q_pos = torch.arange(q_len, device=q.device) + (kv_len - q_len)
        k_pos = torch.arange(kv_len, device=q.device)
        mask = k_pos[None, :] > q_pos[:, None]
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)

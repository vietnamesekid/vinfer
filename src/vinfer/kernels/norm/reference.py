"""Slow, readable RMSNorm variants. The casts follow HF Transformers exactly, since where a value is
rounded to bf16 changes the logits.

Three variants exist across the target models (docs/models/qwen3.md, docs/models/qwen3.5.md):

- ``rms_norm``: Qwen3. Normalize in fp32, cast back to the input dtype, then multiply by ``w``.
- ``rms_norm_offset``: Qwen3.5 layer norms and q/k norms. Multiply by ``1 + w`` in fp32, then
  cast. The weight is stored zero-centered.
- ``rms_norm_gated``: Qwen3.5 Gated DeltaNet output norm. Qwen3-style norm, then multiply by
  ``silu(gate)`` in fp32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_f32(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``weight * normalize(x).to(x.dtype)`` over the last dim."""
    return weight * _normalize_f32(x, eps).to(x.dtype)


def rms_norm_offset(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``(normalize(x) * (1 + weight)).to(x.dtype)``, all in fp32, over the last dim."""
    return (_normalize_f32(x, eps) * (1.0 + weight.float())).to(x.dtype)


def rms_norm_gated(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """``(weight * normalize(x).to(x.dtype) * silu(gate_f32)).to(x.dtype)`` over the last dim.

    The product ``weight * normalized`` happens in the promoted dtype of the two (fp32 when the
    weight is stored in fp32, as in the Qwen3.5 checkpoint).
    """
    y = weight * _normalize_f32(x, eps).to(x.dtype)
    return (y * F.silu(gate.float())).to(x.dtype)

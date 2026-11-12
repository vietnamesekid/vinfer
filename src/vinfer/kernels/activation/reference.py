"""Slow, readable activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up``, in the input dtype (the SwiGLU MLP of every Qwen model here)."""
    return F.silu(gate) * up

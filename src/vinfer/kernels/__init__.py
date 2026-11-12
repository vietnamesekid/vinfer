"""Hot ops. Each has a slow `reference` implementation that optimized versions are tested against.

Must not: change results beyond the parity thresholds.

Op families with interchangeable implementations (attention, Gated DeltaNet) have a backend
registry; ``resolve_kernels`` picks one per family from ``KernelConfig`` for a device and dtype,
once, when the engine is built. Small ops (norms, RoPE, activations) are plain functions until a
profile shows they need backends too.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vinfer.config import KernelConfig
from vinfer.kernels.attention import ATTENTION, AttentionBackend
from vinfer.kernels.gdn import GDN, GDNBackend
from vinfer.kernels.registry import BackendContext


@dataclass(frozen=True)
class Kernels:
    attention: AttentionBackend
    gdn: GDNBackend


def resolve_kernels(cfg: KernelConfig, device: torch.device, dtype: torch.dtype) -> Kernels:
    ctx = BackendContext(torch.device(device), dtype)
    return Kernels(
        attention=ATTENTION.resolve(cfg.attention_backend, ctx),
        gdn=GDN.resolve(cfg.gdn_backend, ctx),
    )


__all__ = ["BackendContext", "Kernels", "resolve_kernels"]

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


@dataclass(frozen=True)
class Kernels:
    attention: AttentionBackend
    gdn: GDNBackend


def resolve_kernels(cfg: KernelConfig, device: torch.device, dtype: torch.dtype) -> Kernels:
    return Kernels(
        attention=ATTENTION.resolve(cfg.attention_backend, device, dtype),
        gdn=GDN.resolve(cfg.gdn_backend, device, dtype),
    )


__all__ = ["Kernels", "resolve_kernels"]

"""Gated DeltaNet: the ``GDNBackend`` interface, its registry, and the reference."""

from vinfer.kernels.gdn.base import GDN, GDNBackend
from vinfer.kernels.gdn.reference import (
    ReferenceGDN,
    causal_conv1d,
    chunk_gated_delta_rule,
    gdn_gating,
    l2norm,
    recurrent_gated_delta_rule,
)

__all__ = [
    "GDN",
    "GDNBackend",
    "ReferenceGDN",
    "causal_conv1d",
    "chunk_gated_delta_rule",
    "gdn_gating",
    "l2norm",
    "recurrent_gated_delta_rule",
]

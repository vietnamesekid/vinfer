"""Gated DeltaNet (linear attention with a fixed-size recurrent state)."""

from vinfer.kernels.gdn.reference import (
    causal_conv1d,
    chunk_gated_delta_rule,
    gdn_gating,
    l2norm,
    recurrent_gated_delta_rule,
)

__all__ = [
    "causal_conv1d",
    "chunk_gated_delta_rule",
    "gdn_gating",
    "l2norm",
    "recurrent_gated_delta_rule",
]

"""RMSNorm variants."""

from vinfer.kernels.norm.reference import rms_norm, rms_norm_gated, rms_norm_offset

__all__ = ["rms_norm", "rms_norm_gated", "rms_norm_offset"]

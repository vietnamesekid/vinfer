"""Attention kernels: the ``AttentionBackend`` interface, its registry, and the reference."""

from vinfer.kernels.attention.base import ATTENTION, AttentionBackend
from vinfer.kernels.attention.reference import ReferenceAttention, attention_reference

__all__ = ["ATTENTION", "AttentionBackend", "ReferenceAttention", "attention_reference"]

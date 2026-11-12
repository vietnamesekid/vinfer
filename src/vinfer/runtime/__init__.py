"""Prefill/decode loop, sampler, stop conditions.

Must not: allocate cache memory directly (ask the state manager).
"""

from vinfer.runtime.generate import Step, generate
from vinfer.runtime.sampler import SamplingParams, sample

__all__ = ["SamplingParams", "Step", "generate", "sample"]

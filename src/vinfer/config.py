"""Every engine and optimization flag lives here, and nowhere else.

Rule: every optimization is a flag here, **off by default** until it passes parity. Each
flag's comment names the experiment that introduced it (``experiments/EXP-NNN-*``), so its
default can be traced back to a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class KernelConfig:
    # "reference" = slow, readable implementation used as ground truth.
    attention_backend: Literal["reference"] = "reference"


@dataclass(frozen=True)
class StateConfig:
    """KV cache and recurrent state (see ``vinfer.state``)."""

    # Optimization flags are added here by their experiments (paged KV, prefix cache, ...).


@dataclass(frozen=True)
class SchedulerConfig:
    """Batching and scheduling policy."""

    # Optimization flags are added here by their experiments (continuous batching, ...).


@dataclass(frozen=True)
class EngineConfig:
    model: str  # HF model id or local path. No default: always explicit.
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    device: str = "cuda"
    seed: int = 0
    kernels: KernelConfig = field(default_factory=KernelConfig)
    state: StateConfig = field(default_factory=StateConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

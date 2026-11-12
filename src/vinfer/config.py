"""Every engine and optimization flag lives here, and nowhere else.

Rule: every optimization is a flag here, **off by default** until it passes parity. Each
flag's comment names the experiment that introduced it (``experiments/EXP-NNN-*``), so its
default can be traced back to a measurement.

``EngineConfig`` is also the single source of truth for the device and dtype: the engine is
built from it (``vinfer.engine.build_engine``) and every component asks the model, never its own
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal

import torch

Dtype = Literal["float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class KernelConfig:
    """Which implementation of each hot op family to use (``vinfer.kernels``). A backend that
    does not support the device or dtype is an error at startup, not a fallback."""

    # "reference" = slow, readable implementation used as ground truth.
    attention_backend: Literal["reference"] = "reference"
    gdn_backend: Literal["reference"] = "reference"

    @classmethod
    def from_string(cls, spec: str | None) -> KernelConfig:
        """``"attention_backend=x,gdn_backend=y"`` (as on command lines); empty = defaults."""
        values: dict[str, str] = {}
        for item in filter(None, (spec or "").split(",")):
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"expected key=value in kernel spec, got {item!r}")
            values[key.strip()] = value.strip()
        known = {f.name for f in fields(cls)}
        if unknown := set(values) - known:
            raise ValueError(f"unknown kernel setting(s) {sorted(unknown)}; known: {sorted(known)}")
        return cls(**values)  # type: ignore[arg-type]

    def describe(self) -> str:
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


@dataclass(frozen=True)
class StateConfig:
    """KV cache and recurrent state (see ``vinfer.state``)."""

    num_slots: int = 1  # sequences held at once
    max_seq_len: int = 8192  # prompt + generated tokens per slot

    # Optimization flags are added here by their experiments (paged KV, prefix cache, ...).


@dataclass(frozen=True)
class SchedulerConfig:
    """Batching and scheduling policy."""

    # Optimization flags are added here by their experiments (continuous batching, ...).


@dataclass(frozen=True)
class EngineConfig:
    model: str  # local checkpoint directory. No default: always explicit.
    dtype: Dtype = "bfloat16"
    device: str = "cuda"
    seed: int = 0
    kernels: KernelConfig = field(default_factory=KernelConfig)
    state: StateConfig = field(default_factory=StateConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

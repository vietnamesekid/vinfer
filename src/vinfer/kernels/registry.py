"""Named implementations of one op family, each declaring the devices and dtypes it supports.

A family (attention, Gated DeltaNet, ...) has one registry. Adding a faster path, say a CUDA
kernel, is a new class registered under a new name plus a value of its flag in
``vinfer.config.KernelConfig``; callers do not change, and no device check is spread through the
model. Choosing a backend the device or dtype does not support fails when the engine is built,
never by silently falling back.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class _Entry(Generic[T]):
    factory: Callable[[], T]
    devices: frozenset[str] | None  # device types ("cpu", "cuda"); None = any
    dtypes: frozenset[torch.dtype] | None  # None = any


class BackendRegistry(Generic[T]):
    def __init__(self, family: str) -> None:
        self.family = family
        self._entries: dict[str, _Entry[T]] = {}

    def register(
        self,
        name: str,
        *,
        devices: set[str] | None = None,
        dtypes: set[torch.dtype] | None = None,
    ) -> Callable[[Callable[[], T]], Callable[[], T]]:
        """Class decorator: register ``cls`` (constructed with no arguments) as ``name``."""

        def wrap(factory: Callable[[], T]) -> Callable[[], T]:
            if name in self._entries:
                raise ValueError(f"{self.family} backend {name!r} registered twice")
            self._entries[name] = _Entry(
                factory,
                frozenset(devices) if devices is not None else None,
                frozenset(dtypes) if dtypes is not None else None,
            )
            return factory

        return wrap

    def names(self) -> list[str]:
        return sorted(self._entries)

    def resolve(self, name: str, device: torch.device, dtype: torch.dtype) -> T:
        if name not in self._entries:
            raise ValueError(f"unknown {self.family} backend {name!r}; available: {self.names()}")
        entry = self._entries[name]
        if entry.devices is not None and device.type not in entry.devices:
            raise ValueError(
                f"{self.family} backend {name!r} does not run on {device.type} "
                f"(supports {sorted(entry.devices)})"
            )
        if entry.dtypes is not None and dtype not in entry.dtypes:
            raise ValueError(
                f"{self.family} backend {name!r} does not support {dtype} "
                f"(supports {sorted(str(d) for d in entry.dtypes)})"
            )
        return entry.factory()

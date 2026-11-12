"""Named implementations of one op family, each declaring where it can run.

A family (attention, Gated DeltaNet, ...) has one registry. Adding a faster path, say a CUDA
kernel, is a new class registered under a new name plus a value of its flag in
``vinfer.config.KernelConfig``; callers do not change, and no device check is spread through the
model. Choosing a backend that cannot run in the given context fails when the engine is built,
never by silently falling back.

A backend declares support by device types and dtypes, plus an optional ``supports`` predicate
for anything finer (compute capability, an installed package). It is constructed with the
``BackendContext``, so it can set up per-device resources (workspaces) at build time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class BackendContext:
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class _Entry(Generic[T]):
    factory: Callable[[BackendContext], T]
    devices: frozenset[str] | None  # device types ("cpu", "cuda"); None = any
    dtypes: frozenset[torch.dtype] | None  # None = any
    supports: Callable[[BackendContext], str | None] | None  # None, or why it cannot run

    def unsupported(self, ctx: BackendContext) -> str | None:
        if self.devices is not None and ctx.device.type not in self.devices:
            return f"does not run on {ctx.device.type} (supports {sorted(self.devices)})"
        if self.dtypes is not None and ctx.dtype not in self.dtypes:
            return f"does not support {ctx.dtype} (supports {sorted(map(str, self.dtypes))})"
        return self.supports(ctx) if self.supports is not None else None


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
        supports: Callable[[BackendContext], str | None] | None = None,
    ) -> Callable[[Callable[[BackendContext], T]], Callable[[BackendContext], T]]:
        """Class decorator: register ``cls`` (constructed as ``cls(ctx)``) as ``name``.
        ``supports(ctx)`` returns ``None`` if the backend can run, else the reason it cannot."""

        def wrap(factory: Callable[[BackendContext], T]) -> Callable[[BackendContext], T]:
            if name in self._entries:
                raise ValueError(f"{self.family} backend {name!r} registered twice")
            self._entries[name] = _Entry(
                factory,
                frozenset(devices) if devices is not None else None,
                frozenset(dtypes) if dtypes is not None else None,
                supports,
            )
            return factory

        return wrap

    def names(self) -> list[str]:
        return sorted(self._entries)

    def unsupported(self, name: str, ctx: BackendContext) -> str | None:
        """Why ``name`` cannot run in ``ctx``, or ``None`` if it can."""
        if name not in self._entries:
            return f"unknown {self.family} backend {name!r}; available: {self.names()}"
        return self._entries[name].unsupported(ctx)

    def resolve(self, name: str, ctx: BackendContext) -> T:
        if (reason := self.unsupported(name, ctx)) is not None:
            prefix = "" if name not in self._entries else f"{self.family} backend {name!r} "
            raise ValueError(prefix + reason)
        return self._entries[name].factory(ctx)

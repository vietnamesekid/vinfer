"""Model architectures and weight loading.

Must not: hold state between calls (state lives in the state manager).

``load_model`` picks the loader from the checkpoint's ``model_type``; adding a model family is
one entry in ``LOADERS``. It resolves the kernel backends once, for every family, so a loader
receives ready ``Kernels`` and never chooses or defaults them itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch

from vinfer.config import KernelConfig
from vinfer.kernels import Kernels, resolve_kernels
from vinfer.models.base import CausalLM

Loader = Callable[[Path, torch.dtype, torch.device, Kernels], CausalLM]


def _qwen(path: Path, dtype: torch.dtype, device: torch.device, kernels: Kernels) -> CausalLM:
    from vinfer.models.qwen import load_qwen

    return load_qwen(path, dtype, device, kernels)


LOADERS: dict[str, Loader] = {"qwen3": _qwen, "qwen3_5": _qwen, "qwen3_5_text": _qwen}


def read_model_type(path: str | Path) -> str | None:
    """The checkpoint's ``model_type`` (top level, else ``text_config``), from ``config.json``."""
    raw = json.loads((Path(path) / "config.json").read_text())
    return raw.get("model_type") or raw.get("text_config", {}).get("model_type")


def load_model(
    path: str | Path,
    dtype: torch.dtype,
    device: str | torch.device,
    kernels: KernelConfig,
) -> CausalLM:
    """Load the checkpoint at ``path`` with the kernel backends ``kernels`` names, which fail
    here if they cannot run on ``device`` in ``dtype``."""
    path, device = Path(path), torch.device(device)
    model_type = read_model_type(path)
    if model_type not in LOADERS:
        raise ValueError(f"unsupported model_type {model_type!r}; supported: {sorted(LOADERS)}")
    return LOADERS[model_type](path, dtype, device, resolve_kernels(kernels, device, dtype))


__all__ = ["LOADERS", "CausalLM", "load_model", "read_model_type"]

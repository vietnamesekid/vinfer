"""Model architectures and weight loading.

Must not: hold state between calls (state lives in the state manager).

``load_model`` picks the loader from the checkpoint's ``model_type``; adding a model family is
one entry in ``LOADERS``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch

from vinfer.models.base import CausalLM

Loader = Callable[[Path, torch.dtype, torch.device], CausalLM]


def _qwen(path: Path, dtype: torch.dtype, device: torch.device) -> CausalLM:
    from vinfer.models.qwen import load_qwen

    return load_qwen(path, dtype, device)


LOADERS: dict[str, Loader] = {"qwen3": _qwen, "qwen3_5": _qwen, "qwen3_5_text": _qwen}


def load_model(path: str | Path, dtype: torch.dtype, device: str | torch.device) -> CausalLM:
    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    model_type = raw.get("model_type") or raw.get("text_config", {}).get("model_type")
    if model_type not in LOADERS:
        raise ValueError(f"unsupported model_type {model_type!r}; supported: {sorted(LOADERS)}")
    return LOADERS[model_type](path, dtype, torch.device(device))


__all__ = ["LOADERS", "CausalLM", "load_model"]

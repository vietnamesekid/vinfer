"""Load a Qwen3 / Qwen3.5 checkpoint (local directory with ``config.json`` and safetensors).

Tensor names are mapped by dropping the text-model prefix (``model.language_model.`` for the
multimodal Qwen3.5 checkpoints, ``model.`` otherwise). Vision (``model.visual.*``) and MTP
(``mtp.*``) weights are skipped. Every remaining tensor must match a parameter in name and shape,
and every parameter must be loaded: nothing is silently left uninitialized or dropped.

Dtypes: tensors are cast to the model dtype, except those the checkpoint stores in fp32 *and*
that the reference keeps in fp32 (``FP32_PARAMS``): Qwen3.5 stores the DeltaNet ``A_log`` and
output-norm weight in fp32 inside a bf16 checkpoint.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from vinfer.kernels import Kernels
from vinfer.kernels.rope import rope_inv_freq
from vinfer.models.qwen.config import QwenConfig
from vinfer.models.qwen.model import QwenModel

SKIP = re.compile(r"^(model\.visual\.|mtp\.|model\.mtp\.)")
PREFIXES = ("model.language_model.", "model.")
FP32_PARAMS = re.compile(r"linear_attn\.(A_log|norm\.weight)$")


def checkpoint_files(path: Path) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [path / n for n in names]
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {path}")
    return files


def internal_name(name: str) -> str | None:
    """Checkpoint tensor name -> model parameter name, or ``None`` to skip it."""
    if SKIP.match(name):
        return None
    if name == "lm_head.weight":
        return name
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    raise ValueError(f"unexpected tensor name {name!r}")


def load_qwen(
    path: str | Path,
    dtype: torch.dtype,
    device: str | torch.device,
    kernels: Kernels,
) -> QwenModel:
    """Load the checkpoint at ``path`` onto ``device``; activations and caches in ``dtype``;
    hot ops from ``kernels`` (resolved by ``vinfer.models.load_model``)."""
    path, device = Path(path), torch.device(device)
    cfg = QwenConfig.from_hf(json.loads((path / "config.json").read_text()))
    with torch.device("meta"):
        model = QwenModel(cfg, kernels, dtype=dtype)
    params = dict(model.named_parameters())
    tensors: dict[str, torch.Tensor] = {}
    lm_head: torch.Tensor | None = None
    for file in checkpoint_files(path):
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                key = internal_name(name)
                if key is None:
                    continue
                t = f.get_tensor(name)
                if key == "lm_head.weight" and cfg.tie_word_embeddings:
                    lm_head = t  # checked against the embedding below
                    continue
                if key not in params:
                    raise ValueError(f"checkpoint tensor {name!r} has no parameter {key!r}")
                if tuple(t.shape) != tuple(params[key].shape):
                    raise ValueError(
                        f"{name}: shape {tuple(t.shape)}, expected {tuple(params[key].shape)}"
                    )
                keep_fp32 = t.dtype == torch.float32 and FP32_PARAMS.search(key)
                tensors[key] = t.to(device=device, dtype=torch.float32 if keep_fp32 else dtype)
    if missing := sorted(set(params) - set(tensors)):
        raise ValueError(f"{len(missing)} parameter(s) not in the checkpoint, e.g. {missing[:5]}")
    if lm_head is not None and not torch.equal(
        lm_head.to(tensors["embed_tokens.weight"].dtype).cpu(), tensors["embed_tokens.weight"].cpu()
    ):
        raise ValueError("tie_word_embeddings is set but lm_head.weight differs from embed_tokens")
    model.load_state_dict(tensors, assign=True, strict=True)
    # Buffers built under the meta device carry no data: recompute them on the target device.
    model.inv_freq = rope_inv_freq(cfg.rotary_dim, cfg.rope_theta).to(device)
    return model.eval().requires_grad_(False)  # inference only

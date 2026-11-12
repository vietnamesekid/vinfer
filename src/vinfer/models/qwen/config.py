"""Model config for Qwen3 and Qwen3.5/3.6 text models, read from a checkpoint's ``config.json``.

See docs/models/qwen3.md and docs/models/qwen3.5.md for what each field means and where it was
checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from vinfer.state import AttentionStateSpec, LayerStateSpec, RecurrentStateSpec

LayerType = Literal["full_attention", "linear_attention"]
SUPPORTED = {"qwen3", "qwen3_5", "qwen3_5_text"}


@dataclass(frozen=True)
class QwenConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    layer_types: tuple[LayerType, ...]
    # Attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rotary_dim: int
    attn_output_gate: bool  # Qwen3.5: q_proj also emits a per-head sigmoid output gate
    zero_centered_norm: bool  # Qwen3.5: RMSNorm multiplies by (1 + w); Qwen3: by w
    # Gated DeltaNet (Qwen3.5 only)
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0
    ssm_dtype: torch.dtype = torch.float32

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    def state_specs(self) -> list[LayerStateSpec]:
        specs: list[LayerStateSpec] = []
        for kind in self.layer_types:
            if kind == "full_attention":
                specs.append(AttentionStateSpec(self.num_key_value_heads, self.head_dim))
            else:
                specs.append(
                    RecurrentStateSpec(
                        conv_dim=self.linear_conv_dim,
                        conv_kernel=self.linear_conv_kernel_dim,
                        num_v_heads=self.linear_num_value_heads,
                        v_head_dim=self.linear_value_head_dim,
                        k_head_dim=self.linear_key_head_dim,
                    )
                )
        return specs

    @classmethod
    def from_hf(cls, raw: dict[str, Any]) -> QwenConfig:
        """From a ``config.json`` dict; the multimodal Qwen3.5 checkpoints nest the text model
        under ``text_config``, and ``tie_word_embeddings`` may sit at either level."""
        text = raw.get("text_config", raw)
        model_type = text.get("model_type", raw.get("model_type"))
        if model_type not in SUPPORTED:
            raise ValueError(f"unsupported model_type {model_type!r}; supported: {SUPPORTED}")
        hybrid = model_type != "qwen3"
        rope = text.get("rope_parameters") or {}
        head_dim = text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]
        rotary_dim = int(head_dim * rope.get("partial_rotary_factor", 1.0))
        num_layers = text["num_hidden_layers"]
        layer_types = tuple(text.get("layer_types") or ["full_attention"] * num_layers)
        if len(layer_types) != num_layers or not set(layer_types) <= {
            "full_attention",
            "linear_attention",
        }:
            raise ValueError(f"bad layer_types {layer_types} for {num_layers} layers")
        if text.get("sliding_window") and text.get("use_sliding_window"):
            raise ValueError("sliding-window attention is not supported")
        scaling = text.get("rope_scaling")
        if rope.get("rope_type", "default") != "default" or (
            scaling and scaling.get("rope_type", scaling.get("type")) != "default"
        ):
            raise ValueError(f"only default RoPE is supported, got {rope or scaling}")
        common = {
            "model_type": model_type,
            "hidden_size": text["hidden_size"],
            "intermediate_size": text["intermediate_size"],
            "vocab_size": text["vocab_size"],
            "rms_norm_eps": text["rms_norm_eps"],
            "tie_word_embeddings": bool(
                text.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))
            ),
            "layer_types": layer_types,
            "num_attention_heads": text["num_attention_heads"],
            "num_key_value_heads": text["num_key_value_heads"],
            "head_dim": head_dim,
            "rope_theta": float(rope.get("rope_theta", text.get("rope_theta", 10000.0))),
            "rotary_dim": rotary_dim,
            "attn_output_gate": bool(text.get("attn_output_gate", hybrid)),
            "zero_centered_norm": hybrid,
        }
        if not hybrid:
            return cls(**common)
        return cls(
            **common,
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            ssm_dtype=getattr(torch, text.get("mamba_ssm_dtype", "float32")),
        )

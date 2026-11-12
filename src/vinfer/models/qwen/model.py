"""Qwen3 and Qwen3.5/3.6 text models (dense; standard or hybrid layers).

Module and parameter names mirror the checkpoint's tensor names without their prefix
(``layers.3.self_attn.q_proj.weight``), so the loader is a rename plus ``load_state_dict``.
The model holds weights only; per-sequence state (KV cache, DeltaNet state) lives in a
``StatePool`` and is passed in on every call.

The hot ops go through the backends in ``Kernels`` (chosen once from ``KernelConfig`` for the
device and dtype): the model never names an implementation or checks a device. Each token mixer
(attention or Gated DeltaNet) takes the same ``StepContext`` and fetches its own state, so the
layer loop has no per-type branches either.

Forward pass and formulas: docs/models/qwen3.md, docs/models/qwen3.5.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vinfer.kernels import Kernels
from vinfer.kernels.activation import swiglu
from vinfer.kernels.attention import AttentionBackend
from vinfer.kernels.gdn import GDNBackend, gdn_gating
from vinfer.kernels.norm import rms_norm, rms_norm_gated, rms_norm_offset
from vinfer.kernels.rope import apply_rope, rope_cos_sin, rope_inv_freq
from vinfer.models.base import CausalLM
from vinfer.models.qwen.config import QwenConfig
from vinfer.state import LayerStateSpec, StatePool


@dataclass(frozen=True)
class StepContext:
    """What every layer needs for one forward call over positions ``start .. start+T-1``."""

    start: int
    slot: int
    pool: StatePool
    cos: torch.Tensor  # [T, rotary_dim]
    sin: torch.Tensor


def _linear(in_features: int, out_features: int) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, zero_centered: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self.zero_centered = zero_centered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = rms_norm_offset if self.zero_centered else rms_norm
        return norm(x, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, cfg: QwenConfig) -> None:
        super().__init__()
        self.gate_proj = _linear(cfg.hidden_size, cfg.intermediate_size)
        self.up_proj = _linear(cfg.hidden_size, cfg.intermediate_size)
        self.down_proj = _linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    """Causal GQA attention with q/k norm, (partial) NeoX RoPE and, for Qwen3.5, a per-head
    sigmoid output gate packed into ``q_proj`` as ``[q_h | gate_h]`` per head."""

    def __init__(self, cfg: QwenConfig, layer_idx: int, backend: AttentionBackend) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.backend = backend
        q_out = cfg.num_attention_heads * cfg.head_dim * (2 if cfg.attn_output_gate else 1)
        self.q_proj = _linear(cfg.hidden_size, q_out)
        self.k_proj = _linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
        self.v_proj = _linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
        self.o_proj = _linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, cfg.zero_centered_norm)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, cfg.zero_centered_norm)

    def forward(self, x: torch.Tensor, ctx: StepContext) -> torch.Tensor:
        cfg, seq_len = self.cfg, x.shape[0]
        q = self.q_proj(x).view(seq_len, cfg.num_attention_heads, -1)
        gate = None
        if cfg.attn_output_gate:  # a model property, fixed per checkpoint
            q, gate = q.chunk(2, dim=-1)
            gate = gate.reshape(seq_len, -1)
        q = self.q_norm(q).transpose(0, 1)  # [heads, T, head_dim]
        k = self.k_norm(self.k_proj(x).view(seq_len, cfg.num_key_value_heads, -1)).transpose(0, 1)
        v = self.v_proj(x).view(seq_len, cfg.num_key_value_heads, -1).transpose(0, 1)
        q, k = apply_rope(q, ctx.cos, ctx.sin), apply_rope(k, ctx.cos, ctx.sin)
        kv_cache = ctx.pool.kv(self.layer_idx, ctx.slot)
        if ctx.start + seq_len > kv_cache[0].shape[1]:
            raise ValueError(
                f"sequence of {ctx.start + seq_len} tokens exceeds the KV capacity "
                f"{kv_cache[0].shape[1]}"
            )
        out = self.backend.forward(q, k, v, kv_cache, ctx.start)
        out = out.transpose(0, 1).reshape(seq_len, -1)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out)


class Conv1dWeight(nn.Module):
    """Holds the depthwise conv weight as the checkpoint stores it, ``[C, 1, kernel]``."""

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel))


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return rms_norm_gated(x, gate, self.weight, self.eps)


class GatedDeltaNet(nn.Module):
    """Linear-attention layer with a fixed-size recurrent state (docs/models/qwen3.5.md)."""

    def __init__(self, cfg: QwenConfig, layer_idx: int, backend: GDNBackend) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.backend = backend
        hv = cfg.linear_num_value_heads
        self.in_proj_qkv = _linear(cfg.hidden_size, cfg.linear_conv_dim)
        self.in_proj_z = _linear(cfg.hidden_size, cfg.linear_value_dim)
        self.in_proj_b = _linear(cfg.hidden_size, hv)
        self.in_proj_a = _linear(cfg.hidden_size, hv)
        self.conv1d = Conv1dWeight(cfg.linear_conv_dim, cfg.linear_conv_kernel_dim)
        self.dt_bias = nn.Parameter(torch.empty(hv))
        self.A_log = nn.Parameter(torch.empty(hv))
        self.norm = GatedRMSNorm(cfg.linear_value_head_dim, cfg.rms_norm_eps)
        self.out_proj = _linear(cfg.linear_value_dim, cfg.hidden_size)

    def forward(self, x: torch.Tensor, ctx: StepContext) -> torch.Tensor:
        cfg, seq_len = self.cfg, x.shape[0]
        conv_state = ctx.pool.conv_state(self.layer_idx, ctx.slot)
        ssm_state = ctx.pool.ssm_state(self.layer_idx, ctx.slot)
        qkv = self.backend.conv(self.in_proj_qkv(x), self.conv1d.weight.squeeze(1), conv_state)
        q, k, v = qkv.split([cfg.linear_key_dim, cfg.linear_key_dim, cfg.linear_value_dim], -1)
        q = q.reshape(seq_len, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        k = k.reshape(seq_len, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
        v = v.reshape(seq_len, cfg.linear_num_value_heads, cfg.linear_value_head_dim)
        g, beta = gdn_gating(self.in_proj_a(x), self.in_proj_b(x), self.A_log, self.dt_bias)
        out = self.backend.rule(q, k, v, g, beta, ssm_state)
        z = self.in_proj_z(x).reshape(seq_len, cfg.linear_num_value_heads, -1)
        out = self.norm(out, z).reshape(seq_len, -1)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: QwenConfig, layer_idx: int, kernels: Kernels) -> None:
        super().__init__()
        # Attribute names follow the checkpoint (self_attn / linear_attn); the mixer is picked
        # here, once, so forward has no per-type branch.
        if cfg.layer_types[layer_idx] == "full_attention":
            self.self_attn = Attention(cfg, layer_idx, kernels.attention)
            self._mixer = "self_attn"
        else:
            self.linear_attn = GatedDeltaNet(cfg, layer_idx, kernels.gdn)
            self._mixer = "linear_attn"
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.zero_centered_norm)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, cfg.zero_centered_norm
        )

    def forward(self, x: torch.Tensor, ctx: StepContext) -> torch.Tensor:
        x = x + getattr(self, self._mixer)(self.input_layernorm(x), ctx)
        return x + self.mlp(self.post_attention_layernorm(x))


class QwenModel(CausalLM):
    def __init__(
        self, cfg: QwenConfig, kernels: Kernels, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__(dtype, device)
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, i, kernels) for i in range(len(cfg.layer_types))
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.zero_centered_norm)
        self.lm_head = None if cfg.tie_word_embeddings else _linear(cfg.hidden_size, cfg.vocab_size)
        self.register_buffer(
            "inv_freq", rope_inv_freq(cfg.rotary_dim, cfg.rope_theta), persistent=False
        )

    def state_specs(self) -> list[LayerStateSpec]:
        return self.cfg.state_specs()

    @property
    def ssm_dtype(self) -> torch.dtype:
        return self.cfg.ssm_dtype

    @torch.inference_mode()
    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        start: int,
        slot: int,
        pool: StatePool,
        layer_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Final-norm hidden states ``[T, hidden]`` for ``input_ids`` (``[T]``) at positions
        ``start .. start+T-1`` of ``slot``. The slot must hold exactly the ``start`` tokens
        before them; its state advances to ``start + T``.

        ``layer_outputs``, if given, receives each decoder layer's output (for debugging parity
        layer by layer)."""
        x = self.embed_tokens(input_ids)
        positions = torch.arange(start, start + input_ids.shape[0], device=input_ids.device)
        cos, sin = rope_cos_sin(positions, self.inv_freq, x.dtype)
        ctx = StepContext(start=start, slot=slot, pool=pool, cos=cos, sin=sin)
        for layer in self.layers:
            x = layer(x, ctx)
            if layer_outputs is not None:
                layer_outputs.append(x.clone())
        return self.norm(x)

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, weight)

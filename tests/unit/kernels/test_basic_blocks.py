"""Basic blocks against HF Transformers' own modules (the correctness reference), on CPU.

The reference kernels copy HF's casts, so the results must be bit-identical, in fp32 and bf16.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vinfer.kernels.activation import swiglu
from vinfer.kernels.norm import rms_norm, rms_norm_gated, rms_norm_offset
from vinfer.kernels.rope import apply_rope, rope_cos_sin, rope_inv_freq

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3 import modeling_qwen3 as hf3  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf35  # noqa: E402

DTYPES = [torch.float32, torch.bfloat16]


def _x(*shape, dtype):
    torch.manual_seed(0)
    return (torch.randn(*shape) * 3).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_matches_qwen3(dtype):
    x = _x(2, 5, 64, dtype=dtype)
    hf = hf3.Qwen3RMSNorm(64, eps=1e-6).to(dtype)
    with torch.no_grad():
        hf.weight.copy_(torch.randn(64))
    assert torch.equal(rms_norm(x, hf.weight.detach(), 1e-6), hf(x).detach())


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_offset_matches_qwen3_5(dtype):
    x = _x(2, 5, 64, dtype=dtype)
    hf = hf35.Qwen3_5RMSNorm(64, eps=1e-6).to(dtype)
    with torch.no_grad():
        hf.weight.copy_(torch.randn(64) * 0.1)
    assert torch.equal(rms_norm_offset(x, hf.weight.detach(), 1e-6), hf(x).detach())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weight_dtype", [None, torch.float32])
def test_rms_norm_gated_matches_qwen3_5(dtype, weight_dtype):
    # weight_dtype fp32: the checkpoint stores this weight in fp32 even in a bf16 model.
    x, z = _x(6, 32, dtype=dtype), _x(6, 32, dtype=dtype).flip(0)
    hf = hf35.Qwen3_5RMSNormGated(32, eps=1e-6).to(weight_dtype or dtype)
    with torch.no_grad():
        hf.weight.copy_(torch.randn(32))
    assert torch.equal(rms_norm_gated(x, z, hf.weight.detach(), 1e-6), hf(x, z).detach())


@pytest.mark.parametrize("dtype", DTYPES)
def test_full_rope_matches_qwen3(dtype):
    heads, seq, head_dim, theta = 4, 7, 32, 1e6
    q, k = _x(1, heads, seq, head_dim, dtype=dtype), _x(1, heads, seq, head_dim, dtype=dtype) * -1
    positions = torch.arange(3, 3 + seq)[None]
    cfg = transformers.Qwen3Config(head_dim=head_dim, rope_theta=theta, num_attention_heads=heads)
    hf_cos, hf_sin = hf3.Qwen3RotaryEmbedding(cfg)(q, positions)
    cos, sin = rope_cos_sin(positions, rope_inv_freq(head_dim, theta), dtype)
    assert torch.equal(cos, hf_cos) and torch.equal(sin, hf_sin)
    hf_q, hf_k = hf3.apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    assert torch.equal(apply_rope(q, cos[:, None], sin[:, None]), hf_q)
    assert torch.equal(apply_rope(k, cos[:, None], sin[:, None]), hf_k)


@pytest.mark.parametrize("dtype", DTYPES)
def test_partial_rope_matches_qwen3_5_text_mrope(dtype):
    # Text only: the three MRoPE position rows are equal, so HF's interleaved MRoPE must reduce
    # to plain 1-D RoPE on the first head_dim * partial_rotary_factor dims.
    heads, seq, head_dim, theta, factor = 2, 9, 64, 1e7, 0.25
    rotary_dim = int(head_dim * factor)
    q = _x(1, heads, seq, head_dim, dtype=dtype)
    positions = torch.arange(seq)[None]
    cfg = transformers.Qwen3_5TextConfig(
        head_dim=head_dim,
        num_attention_heads=heads,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": theta,
            "partial_rotary_factor": factor,
            "mrope_interleaved": True,
            "mrope_section": [3, 3, 2],
        },
    )
    hf_cos, hf_sin = hf35.Qwen3_5TextRotaryEmbedding(cfg)(q, positions.expand(3, 1, -1))
    cos, sin = rope_cos_sin(positions, rope_inv_freq(rotary_dim, theta), dtype)
    assert torch.equal(cos, hf_cos) and torch.equal(sin, hf_sin)
    hf_q, _ = hf35.apply_rotary_pos_emb(q, q, hf_cos, hf_sin)
    out = apply_rope(q, cos[:, None], sin[:, None])
    assert torch.equal(out, hf_q)
    assert torch.equal(out[..., rotary_dim:], q[..., rotary_dim:])  # pass-through dims untouched


def test_rope_rejects_odd_rotary_dim():
    with pytest.raises(ValueError):
        rope_inv_freq(7, 1e4)


@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_matches_hf_mlp(dtype):
    cfg = transformers.Qwen3Config(hidden_size=16, intermediate_size=24, hidden_act="silu")
    mlp = hf3.Qwen3MLP(cfg).to(dtype)
    x = _x(3, 16, dtype=dtype)
    with torch.no_grad():
        ours = F.linear(
            swiglu(F.linear(x, mlp.gate_proj.weight), F.linear(x, mlp.up_proj.weight)),
            mlp.down_proj.weight,
        )
        assert torch.equal(ours, mlp(x))

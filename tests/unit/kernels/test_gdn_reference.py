"""Gated DeltaNet reference: the two forms against each other and against HF Transformers, on CPU.

Shapes use K != V and HV != H on purpose: Qwen3.5-2B has K == V and HV == H, where a transposed
state or a wrong key-head mapping still runs and is silently wrong.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vinfer.kernels.gdn import (
    causal_conv1d,
    chunk_gated_delta_rule,
    gdn_gating,
    recurrent_gated_delta_rule,
)

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf  # noqa: E402

B, H, HV, K, V = 2, 2, 6, 8, 12
TOL = {"atol": 2e-5, "rtol": 1e-4}


def _inputs(seq_len: int, seed: int = 0, heads: tuple[int, int] = (H, HV)):
    h, hv = heads
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, seq_len, h, K, generator=gen)
    k = torch.randn(B, seq_len, h, K, generator=gen)
    v = torch.randn(B, seq_len, hv, V, generator=gen)
    # Realistic gates: g in about [-2, 0) (decay), beta in (0, 1).
    g, beta = gdn_gating(
        torch.randn(B, seq_len, hv, generator=gen),
        torch.randn(B, seq_len, hv, generator=gen),
        torch.rand(hv, generator=gen).log(),
        torch.randn(hv, generator=gen),
    )
    return q, k, v, g, beta


def _state(seed: int = 1, hv: int = HV):
    return torch.randn(B, hv, V, K, generator=torch.Generator().manual_seed(seed)) * 0.1


def _split(tensors, start, end):
    return [x[:, start:end] for x in tensors]


@pytest.mark.parametrize("with_state", [False, True])
def test_recurrent_matches_hf(with_state):
    q, k, v, g, beta = _inputs(9)
    state = _state() if with_state else None
    out, final = recurrent_gated_delta_rule(q, k, v, g, beta, state)
    # HF wants q/k already repeated to HV heads, and its state is [K, V] (ours is [V, K]).
    r = HV // H
    hf_out, hf_final = hf.torch_recurrent_gated_delta_rule(
        q.repeat_interleave(r, 2),
        k.repeat_interleave(r, 2),
        v,
        g=g,
        beta=beta,
        initial_state=None if state is None else state.transpose(-1, -2),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out, hf_out, **TOL)
    torch.testing.assert_close(final, hf_final.transpose(-1, -2), **TOL)


@pytest.mark.parametrize("seq_len", [1, 5, 15, 16, 17, 33])
@pytest.mark.parametrize("with_state", [False, True])
def test_chunk_equals_recurrent(seq_len, with_state):
    # chunk_size 16 so that 15/16/17/33 cover below, at, above and two blocks plus one.
    inputs = _inputs(seq_len, seed=seq_len)
    state = _state() if with_state else None
    ref, ref_final = recurrent_gated_delta_rule(*inputs, state)
    out, final = chunk_gated_delta_rule(*inputs, state, chunk_size=16)
    torch.testing.assert_close(out, ref, **TOL)
    torch.testing.assert_close(final, ref_final, **TOL)


@pytest.mark.parametrize("seq_len", [63, 64, 65, 129])
def test_chunk_equals_recurrent_at_the_real_chunk_size(seq_len):
    inputs = _inputs(seq_len, seed=seq_len)
    ref, ref_final = recurrent_gated_delta_rule(*inputs)
    out, final = chunk_gated_delta_rule(*inputs)  # default chunk_size=64
    torch.testing.assert_close(out, ref, **TOL)
    torch.testing.assert_close(final, ref_final, **TOL)


def test_chunk_matches_hf_chunk():
    q, k, v, g, beta = _inputs(70)
    state = _state()
    r = HV // H
    out, final = chunk_gated_delta_rule(q, k, v, g, beta, state)
    hf_out, hf_final = hf.torch_chunk_gated_delta_rule(
        q.repeat_interleave(r, 2),
        k.repeat_interleave(r, 2),
        v,
        g=g,
        beta=beta,
        initial_state=state.transpose(-1, -2),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out, hf_out, **TOL)
    torch.testing.assert_close(final, hf_final.transpose(-1, -2), **TOL)


@pytest.mark.parametrize("split", [1, 7, 16, 20])
def test_prefill_in_two_calls_equals_one(split):
    # Second call starts its blocks at 0 again; carrying the state must still give the same.
    inputs = _inputs(37, seed=3)
    whole, whole_final = chunk_gated_delta_rule(*inputs, chunk_size=16)
    first, mid = chunk_gated_delta_rule(*_split(inputs, 0, split), chunk_size=16)
    second, final = chunk_gated_delta_rule(*_split(inputs, split, 37), mid, chunk_size=16)
    torch.testing.assert_close(torch.cat([first, second], 1), whole, **TOL)
    torch.testing.assert_close(final, whole_final, **TOL)


def test_chunk_prefill_then_recurrent_decode_equals_recurrent_throughout():
    inputs = _inputs(40, seed=4)
    ref, ref_final = recurrent_gated_delta_rule(*inputs)
    prefill, state = chunk_gated_delta_rule(*_split(inputs, 0, 30), chunk_size=16)
    outs = [prefill]
    for t in range(30, 40):
        o, state = recurrent_gated_delta_rule(*_split(inputs, t, t + 1), state)
        outs.append(o)
    torch.testing.assert_close(torch.cat(outs, 1), ref, **TOL)
    torch.testing.assert_close(state, ref_final, **TOL)


def test_value_head_reads_its_own_key_head():
    # With HV = 3H, value heads 0-2 read key head 0 and 3-5 read key head 1 (j // 3, not j % H).
    q, k, v, g, beta = _inputs(6, seed=5)
    out, _ = recurrent_gated_delta_rule(q, k, v, g, beta)
    for j in range(HV):
        kh = j // (HV // H)
        single, _ = recurrent_gated_delta_rule(
            q[:, :, kh : kh + 1],
            k[:, :, kh : kh + 1],
            v[:, :, j : j + 1],
            g[:, :, j : j + 1],
            beta[:, :, j : j + 1],
        )
        torch.testing.assert_close(out[:, :, j : j + 1], single, **TOL)


def test_state_layout_is_value_by_key():
    # One step from zero state with g = 0, beta = 1: S = v ⊗ k_normalized, shape [V, K].
    q, k, v, _, _ = _inputs(1, seed=6, heads=(1, 1))
    zeros, ones = torch.zeros(B, 1, 1), torch.ones(B, 1, 1)
    _, state = recurrent_gated_delta_rule(q, k, v, zeros, ones)
    k_unit = k / k.norm(dim=-1, keepdim=True)
    expected = v[:, 0, 0, :, None] * k_unit[:, 0, 0, None, :]
    torch.testing.assert_close(state[:, 0], expected, atol=1e-5, rtol=1e-4)


def test_initial_state_is_not_modified():
    inputs, state = _inputs(5, seed=7), _state()
    before = state.clone()
    recurrent_gated_delta_rule(*inputs, state)
    chunk_gated_delta_rule(*inputs, state, chunk_size=4)
    assert torch.equal(state, before)


def test_output_dtype_follows_v_and_state_is_fp32():
    q, k, v, g, beta = _inputs(5, seed=8)
    out, state = chunk_gated_delta_rule(q, k, v.bfloat16(), g, beta, chunk_size=4)
    assert out.dtype == torch.bfloat16 and state.dtype == torch.float32


def test_gating_matches_hf_formula():
    gen = torch.Generator().manual_seed(9)
    a, b = torch.randn(3, 4, generator=gen), torch.randn(3, 4, generator=gen)
    a_log, dt_bias = torch.randn(4, generator=gen), torch.randn(4, generator=gen)
    g, beta = gdn_gating(a, b, a_log, dt_bias)
    assert torch.equal(g, -a_log.float().exp() * F.softplus(a.float() + dt_bias))
    assert torch.equal(beta, b.sigmoid()) and bool((g <= 0).all())


# --- short convolution -------------------------------------------------------------------------

C, W = 10, 4


def _conv_inputs(seq_len, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, seq_len, C, generator=gen), torch.randn(C, W, generator=gen)


def test_conv_matches_hf():
    x, weight = _conv_inputs(9)
    y, state = causal_conv1d(x, weight)
    hf_y = hf.causal_conv1d_fn(x.transpose(1, 2), weight, None, activation="silu")
    torch.testing.assert_close(y, hf_y.transpose(1, 2), atol=1e-6, rtol=1e-6)
    assert torch.equal(state, x[:, -(W - 1) :].transpose(1, 2))  # raw inputs, not outputs


@pytest.mark.parametrize("n", range(1, 9))
def test_conv_prefill_equals_prefill_then_one_step(n):
    x, weight = _conv_inputs(n, seed=n)
    whole, whole_state = causal_conv1d(x, weight)
    head, state = causal_conv1d(x[:, : n - 1], weight)
    tail, state = causal_conv1d(x[:, n - 1 :], weight, state)
    torch.testing.assert_close(torch.cat([head, tail], 1), whole, atol=1e-6, rtol=1e-6)
    assert torch.equal(state, whole_state)


def test_conv_state_after_a_prompt_shorter_than_the_kernel():
    # Two tokens < W - 1: the state keeps the older zeros in front, in order.
    x, weight = _conv_inputs(2, seed=11)
    _, state = causal_conv1d(x, weight)
    assert torch.equal(state[..., 0], torch.zeros(B, C))
    assert torch.equal(state[..., 1:], x.transpose(1, 2))

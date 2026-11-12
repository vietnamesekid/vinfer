"""Every registered backend against the reference functions, on this machine's devices, in
fp32 and bf16.

A new backend (say a CUDA kernel) is picked up here automatically: it must match the reference
within tolerance and update state in place. A backend that cannot run in a (device, dtype)
context is skipped there, with its reason; one that runs nowhere available is still reported by
the skip list.
"""

from __future__ import annotations

import pytest
import torch

from vinfer.kernels.attention import ATTENTION, attention_reference
from vinfer.kernels.gdn import GDN, causal_conv1d, chunk_gated_delta_rule, gdn_gating
from vinfer.kernels.registry import BackendContext

DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.bfloat16]
# Inputs are rounded to the backend dtype first and the reference runs on those, so the
# tolerance only covers the backend's own rounding.
TOL = {torch.float32: {"atol": 2e-4, "rtol": 2e-3}, torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2}}


def _backend(registry, name, device, dtype):
    ctx = BackendContext(device, dtype)
    if (reason := registry.unsupported(name, ctx)) is not None:
        pytest.skip(reason)
    return registry.resolve(name, ctx)


def _close(actual, expected, dtype):
    torch.testing.assert_close(actual.float().cpu(), expected.float().cpu(), **TOL[dtype])


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("name", ATTENTION.names())
@pytest.mark.parametrize("start", [0, 5])
def test_attention_backend(name, device, dtype, start):
    backend = _backend(ATTENTION, name, device, dtype)
    hq, hkv, t, d, cap = 4, 2, 3, 8, 16
    g = _gen()
    k_cache, v_cache = (torch.randn(hkv, cap, d, generator=g).to(device, dtype) for _ in range(2))
    q, k, v = (torch.randn(h, t, d, generator=g).to(device, dtype) for h in (hq, hkv, hkv))
    expected_k, expected_v = k_cache.clone(), v_cache.clone()
    expected_k[:, start : start + t], expected_v[:, start : start + t] = k, v
    expected = attention_reference(
        q[None].float(),
        expected_k[None, :, : start + t].float(),
        expected_v[None, :, : start + t].float(),
    )[0]
    out = backend.forward(q, k, v, (k_cache, v_cache), start)
    assert out.dtype == dtype
    _close(out, expected, dtype)
    _close(k_cache[:, : start + t], expected_k[:, : start + t], dtype)
    _close(v_cache[:, : start + t], expected_v[:, : start + t], dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("name", GDN.names())
@pytest.mark.parametrize("seq_len", [1, 5, 70])
def test_gdn_backend_rule(name, device, dtype, seq_len):
    backend = _backend(GDN, name, device, dtype)
    h, hv, kd, vd = 2, 6, 8, 12
    g_ = _gen(seq_len)
    q, k = (torch.randn(seq_len, h, kd, generator=g_).to(dtype) for _ in range(2))
    v = torch.randn(seq_len, hv, vd, generator=g_).to(dtype)
    gate, beta = gdn_gating(
        torch.randn(seq_len, hv, generator=g_),
        torch.randn(seq_len, hv, generator=g_),
        torch.rand(hv, generator=g_).log(),
        torch.randn(hv, generator=g_),
    )
    state = torch.randn(hv, vd, kd, generator=g_) * 0.1  # SSM state stays fp32
    ref_out, ref_state = chunk_gated_delta_rule(
        q[None], k[None], v[None], gate[None], beta[None], state[None]
    )
    state_dev = state.to(device)
    out = backend.rule(*(x.to(device) for x in (q, k, v, gate, beta)), state_dev)
    assert out.dtype == dtype
    _close(out, ref_out[0], dtype)
    _close(state_dev, ref_state[0], dtype)  # updated in place


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("name", GDN.names())
@pytest.mark.parametrize("seq_len", [1, 2, 9])
def test_gdn_backend_conv(name, device, dtype, seq_len):
    backend = _backend(GDN, name, device, dtype)
    g = _gen(seq_len)
    x = torch.randn(seq_len, 10, generator=g).to(dtype)
    weight = torch.randn(10, 4, generator=g).to(dtype)
    state = torch.randn(10, 3, generator=g).to(dtype)
    ref_y, ref_state = causal_conv1d(x[None].float(), weight.float(), state[None].float())
    state_dev = state.to(device)
    y = backend.conv(x.to(device), weight.to(device), state_dev)
    assert y.dtype == dtype
    _close(y, ref_y[0], dtype)
    _close(state_dev, ref_state[0], dtype)  # updated in place

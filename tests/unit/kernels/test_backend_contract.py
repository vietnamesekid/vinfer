"""Every registered backend against the reference functions, on this machine's devices.

A new backend (say a CUDA kernel) is picked up here automatically: it must match the reference
within tolerance and update state in place. Backends that do not run on an available device are
skipped, not failed.
"""

from __future__ import annotations

import pytest
import torch

from vinfer.kernels.attention import ATTENTION, attention_reference
from vinfer.kernels.gdn import GDN, causal_conv1d, chunk_gated_delta_rule, gdn_gating

DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
TOL = {"atol": 2e-4, "rtol": 2e-3}


def _backend(registry, name, device, dtype=torch.float32):
    try:
        return registry.resolve(name, device, dtype)
    except ValueError as exc:
        pytest.skip(str(exc))


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("name", ATTENTION.names())
@pytest.mark.parametrize("start", [0, 5])
def test_attention_backend(name, device, start):
    backend = _backend(ATTENTION, name, device)
    hq, hkv, t, d, cap = 4, 2, 3, 8, 16
    g = _gen()
    k_cache, v_cache = (torch.randn(hkv, cap, d, generator=g).to(device) for _ in range(2))
    q, k, v = (torch.randn(h, t, d, generator=g).to(device) for h in (hq, hkv, hkv))
    expected_k, expected_v = k_cache.clone(), v_cache.clone()
    expected_k[:, start : start + t], expected_v[:, start : start + t] = k, v
    expected = attention_reference(
        q[None], expected_k[None, :, : start + t], expected_v[None, :, : start + t]
    )[0]
    out = backend.forward(q, k, v, (k_cache, v_cache), start)
    torch.testing.assert_close(out, expected, **TOL)
    torch.testing.assert_close(k_cache[:, : start + t], expected_k[:, : start + t], **TOL)
    torch.testing.assert_close(v_cache[:, : start + t], expected_v[:, : start + t], **TOL)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("name", GDN.names())
@pytest.mark.parametrize("seq_len", [1, 5, 70])
def test_gdn_backend_rule(name, device, seq_len):
    backend = _backend(GDN, name, device)
    h, hv, kd, vd = 2, 6, 8, 12
    g_ = _gen(seq_len)
    q, k = (torch.randn(seq_len, h, kd, generator=g_) for _ in range(2))
    v = torch.randn(seq_len, hv, vd, generator=g_)
    gate, beta = gdn_gating(
        torch.randn(seq_len, hv, generator=g_),
        torch.randn(seq_len, hv, generator=g_),
        torch.rand(hv, generator=g_).log(),
        torch.randn(hv, generator=g_),
    )
    state = torch.randn(hv, vd, kd, generator=g_) * 0.1
    ref_out, ref_state = chunk_gated_delta_rule(
        q[None], k[None], v[None], gate[None], beta[None], state[None]
    )
    state_dev = state.to(device)
    out = backend.rule(*(x.to(device) for x in (q, k, v, gate, beta)), state_dev)
    torch.testing.assert_close(out.cpu(), ref_out[0], **TOL)
    torch.testing.assert_close(state_dev.cpu(), ref_state[0], **TOL)  # updated in place


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("name", GDN.names())
@pytest.mark.parametrize("seq_len", [1, 2, 9])
def test_gdn_backend_conv(name, device, seq_len):
    backend = _backend(GDN, name, device)
    g = _gen(seq_len)
    x, weight = torch.randn(seq_len, 10, generator=g), torch.randn(10, 4, generator=g)
    state = torch.randn(10, 3, generator=g)
    ref_y, ref_state = causal_conv1d(x[None], weight, state[None])
    state_dev = state.to(device)
    y = backend.conv(x.to(device), weight.to(device), state_dev)
    torch.testing.assert_close(y.cpu(), ref_y[0], **TOL)
    torch.testing.assert_close(state_dev.cpu(), ref_state[0], **TOL)  # updated in place

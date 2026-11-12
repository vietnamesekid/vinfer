"""StatePool: slot lifecycle, isolation and sizing (CPU)."""

from __future__ import annotations

import pytest
import torch

from vinfer.state import AttentionStateSpec, RecurrentStateSpec, StatePool

ATTN = AttentionStateSpec(num_kv_heads=2, head_dim=4)
GDN = RecurrentStateSpec(conv_dim=6, conv_kernel=4, num_v_heads=3, v_head_dim=5, k_head_dim=2)
LAYERS = [GDN, ATTN]


def _pool(slots=2, max_len=8):
    return StatePool(LAYERS, slots, max_len, dtype=torch.bfloat16, ssm_dtype=torch.float32)


def _fill(pool, slot, value):
    k, v = pool.kv(1, slot)
    k.fill_(value)
    v.fill_(value)
    pool.conv_state(0, slot).fill_(value)
    pool.ssm_state(0, slot).fill_(value)


def _all(pool, slot):
    return [*pool.kv(1, slot), pool.conv_state(0, slot), pool.ssm_state(0, slot)]


def test_shapes_and_dtypes():
    pool = _pool()
    k, _ = pool.kv(1, 0)
    assert k.shape == (2, 8, 4) and k.dtype == torch.bfloat16
    assert pool.conv_state(0, 0).shape == (6, 3) and pool.conv_state(0, 0).dtype == torch.bfloat16
    assert pool.ssm_state(0, 0).shape == (3, 5, 2) and pool.ssm_state(0, 0).dtype == torch.float32


def test_a_slot_is_zero_when_allocated_again():
    pool = _pool(slots=1)
    slot = pool.alloc()
    _fill(pool, slot, 3.0)
    pool.free(slot)
    assert pool.alloc() == slot
    assert all(not t.any() for t in _all(pool, slot))


def test_views_write_into_the_pool_and_slots_are_isolated():
    pool = _pool()
    a, b = pool.alloc(), pool.alloc()
    _fill(pool, a, 1.0)
    assert all(bool((t == 1).all()) for t in _all(pool, a))
    assert all(not t.any() for t in _all(pool, b))


def test_copy_slot_is_an_independent_copy():
    pool = _pool()
    a, b = pool.alloc(), pool.alloc()
    _fill(pool, a, 2.0)
    pool.copy_slot(a, b)
    _fill(pool, a, 5.0)
    assert all(bool((t == 2).all()) for t in _all(pool, b))


def test_exhaustion_double_free_and_range():
    pool = _pool(slots=1)
    slot = pool.alloc()
    with pytest.raises(RuntimeError, match="no free slot"):
        pool.alloc()
    pool.free(slot)
    with pytest.raises(ValueError, match="already free"):
        pool.free(slot)
    with pytest.raises(IndexError):
        pool.reset(5)


def test_bytes_per_slot_matches_the_buffers():
    pool = _pool(slots=1)
    actual = sum(t.numel() * t.element_size() for t in _all(pool, 0))
    assert StatePool.bytes_per_slot(LAYERS, 8, torch.bfloat16, torch.float32) == actual

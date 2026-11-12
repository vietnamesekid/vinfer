"""Allocating a slot must not clear its whole KV cache.

Bug: ``alloc()`` zeroed every attention layer's ``max_seq_len`` positions (about 17 GB of
memset per request for a 27B model at 262K context) although attention only reads positions the
sequence has written. Only the recurrent state needs zeroing.
"""

from __future__ import annotations

import torch

from vinfer.state import AttentionStateSpec, RecurrentStateSpec, StatePool


def test_alloc_zeroes_recurrent_state_only():
    specs = [RecurrentStateSpec(4, 4, 2, 3, 2), AttentionStateSpec(2, 4)]
    pool = StatePool(specs, num_slots=1, max_seq_len=8, dtype=torch.float32)
    slot = pool.alloc()
    k, _ = pool.kv(1, slot)
    k.fill_(7.0)
    pool.ssm_state(0, slot).fill_(7.0)
    pool.free(slot)
    slot = pool.alloc()
    assert not pool.ssm_state(0, slot).any()
    assert bool((pool.kv(1, slot)[0] == 7.0).all())

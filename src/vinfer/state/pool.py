"""Per-slot state pool, version 0: contiguous buffers, one slot per sequence.

Two kinds of state, as the hybrid Qwen models need (docs/architecture.md):

- **Attention layers**: a KV cache per token, ``[slots, kv_heads, max_seq_len, head_dim]`` for K
  and for V, in the model dtype. Contiguous per slot (paging comes later, as an experiment).
- **Gated DeltaNet layers**: fixed-size recurrent state per slot. The short-convolution state
  ``[slots, conv_dim, kernel - 1]`` in the model dtype, and the SSM state ``[slots, HV, V, K]`` in
  ``ssm_dtype`` (fp32 for Qwen3.5, from ``mamba_ssm_dtype``).

The pool knows slots, not requests or token counts: which positions of a slot are filled is the
caller's business. A slot is zeroed when allocated, so a sequence never starts from a previous
sequence's recurrent state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionStateSpec:
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class RecurrentStateSpec:
    conv_dim: int
    conv_kernel: int
    num_v_heads: int
    v_head_dim: int
    k_head_dim: int


LayerStateSpec = AttentionStateSpec | RecurrentStateSpec


class StatePool:
    def __init__(
        self,
        layers: list[LayerStateSpec],
        num_slots: int,
        max_seq_len: int,
        dtype: torch.dtype,
        ssm_dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.layers = layers
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self._k: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}
        self._conv: dict[int, torch.Tensor] = {}
        self._ssm: dict[int, torch.Tensor] = {}
        for i, spec in enumerate(layers):
            if isinstance(spec, AttentionStateSpec):
                shape = (num_slots, spec.num_kv_heads, max_seq_len, spec.head_dim)
                self._k[i] = torch.zeros(shape, dtype=dtype, device=device)
                self._v[i] = torch.zeros(shape, dtype=dtype, device=device)
            else:
                conv = (num_slots, spec.conv_dim, spec.conv_kernel - 1)
                ssm = (num_slots, spec.num_v_heads, spec.v_head_dim, spec.k_head_dim)
                self._conv[i] = torch.zeros(conv, dtype=dtype, device=device)
                self._ssm[i] = torch.zeros(ssm, dtype=ssm_dtype, device=device)
        self._free = list(range(num_slots - 1, -1, -1))  # pop() hands out slot 0 first

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError(f"no free slot (pool has {self.num_slots})")
        slot = self._free.pop()
        self.reset(slot)
        return slot

    def free(self, slot: int) -> None:
        self._check(slot)
        if slot in self._free:
            raise ValueError(f"slot {slot} is already free")
        self._free.append(slot)

    def reset(self, slot: int) -> None:
        """Zero every buffer of ``slot``."""
        self._check(slot)
        for buffers in (self._k, self._v, self._conv, self._ssm):
            for t in buffers.values():
                t[slot].zero_()

    def copy_slot(self, src: int, dst: int) -> None:
        """Make ``dst`` an independent copy of ``src`` (all layers, all positions)."""
        self._check(src)
        self._check(dst)
        for buffers in (self._k, self._v, self._conv, self._ssm):
            for t in buffers.values():
                t[dst].copy_(t[src])

    @property
    def num_free(self) -> int:
        return len(self._free)

    def _check(self, slot: int) -> None:
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range [0, {self.num_slots})")

    def kv(self, layer: int, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(k, v)`` of ``[kv_heads, max_seq_len, head_dim]`` for an attention layer."""
        return self._k[layer][slot], self._v[layer][slot]

    def conv_state(self, layer: int, slot: int) -> torch.Tensor:
        """``[conv_dim, kernel - 1]`` for a Gated DeltaNet layer."""
        return self._conv[layer][slot]

    def ssm_state(self, layer: int, slot: int) -> torch.Tensor:
        """``[HV, V, K]`` for a Gated DeltaNet layer."""
        return self._ssm[layer][slot]

    @staticmethod
    def bytes_per_slot(
        layers: list[LayerStateSpec],
        max_seq_len: int,
        dtype: torch.dtype,
        ssm_dtype: torch.dtype = torch.float32,
    ) -> int:
        """Bytes one slot takes: attention ``2 * kv_heads * max_seq_len * head_dim * size(dtype)``
        per layer, plus ``conv_dim * (kernel - 1) * size(dtype) + HV * V * K * size(ssm_dtype)``
        per Gated DeltaNet layer."""
        size, ssm_size = dtype.itemsize, ssm_dtype.itemsize
        total = 0
        for spec in layers:
            if isinstance(spec, AttentionStateSpec):
                total += 2 * spec.num_kv_heads * max_seq_len * spec.head_dim * size
            else:
                total += spec.conv_dim * (spec.conv_kernel - 1) * size
                total += spec.num_v_heads * spec.v_head_dim * spec.k_head_dim * ssm_size
        return total

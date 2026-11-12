"""The interface every model offers to the rest of the engine.

The runtime, scheduler and parity tools depend on ``CausalLM`` only, never on a model's
internals (layer names, where the embedding lives), so a new model family (MLA, MoE, sharded)
plugs in without touching them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from vinfer.state import LayerStateSpec, StatePool


class CausalLM(nn.Module, ABC):
    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = torch.device(device)

    @property
    def dtype(self) -> torch.dtype:
        """Activation and cache dtype (some weights may be stored in fp32)."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def state_specs(self) -> list[LayerStateSpec]:
        """Per-layer state the model needs from the pool."""

    @property
    @abstractmethod
    def ssm_dtype(self) -> torch.dtype:
        """dtype of recurrent (SSM) state, which may differ from ``dtype``."""

    @abstractmethod
    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        start: int,
        slot: int,
        pool: StatePool,
        layer_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Final hidden states ``[T, hidden]`` for ``input_ids`` at positions
        ``start .. start+T-1`` of ``slot``; the slot's state advances to ``start + T``."""

    @abstractmethod
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[T, vocab]`` logits for hidden states."""

    def new_state_pool(self, num_slots: int, max_seq_len: int) -> StatePool:
        """A pool shaped for this model, on its device and in its dtypes."""
        return StatePool(
            self.state_specs(),
            num_slots=num_slots,
            max_seq_len=max_seq_len,
            dtype=self.dtype,
            ssm_dtype=self.ssm_dtype,
            device=self.device,
        )

"""Build the engine's parts from one ``EngineConfig``: the single place where the device, dtype
and sizes are decided. Everything downstream reads them from the model (``CausalLM.device``,
``CausalLM.dtype``), never from its own default."""

from __future__ import annotations

from dataclasses import dataclass

from vinfer.config import EngineConfig
from vinfer.models import CausalLM, load_model
from vinfer.runtime.tokenizer import Tokenizer
from vinfer.state import StatePool


@dataclass
class Engine:
    config: EngineConfig
    model: CausalLM
    tokenizer: Tokenizer
    pool: StatePool


def build_engine(cfg: EngineConfig) -> Engine:
    model = load_model(cfg.model, cfg.torch_dtype, cfg.torch_device, cfg.kernels)
    pool = model.new_state_pool(cfg.state.num_slots, cfg.state.max_seq_len)
    return Engine(cfg, model, Tokenizer(cfg.model), pool)

"""build_engine / load_model: the device, dtype and sizes come from EngineConfig only (CPU)."""

from __future__ import annotations

import json

import pytest
import torch

from tests.unit.models import fixtures
from vinfer.config import EngineConfig, KernelConfig, StateConfig
from vinfer.engine import build_engine
from vinfer.models import CausalLM, load_model, read_model_type

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = fixtures.save(fixtures.hf_qwen3_5(), tmp_path_factory.mktemp("eng"))
    return fixtures.save_tiny_tokenizer(path)


def test_engine_parts_follow_the_config(checkpoint):
    cfg = EngineConfig(
        model=str(checkpoint),
        dtype="bfloat16",
        device="cpu",
        state=StateConfig(num_slots=2, max_seq_len=40),
    )
    engine = build_engine(cfg)
    assert isinstance(engine.model, CausalLM)
    assert engine.model.device == torch.device("cpu") and engine.model.dtype == torch.bfloat16
    assert engine.pool.num_slots == 2 and engine.pool.max_seq_len == 40
    k, _ = engine.pool.kv(1, 0)  # layer 1 is full attention in the tiny config
    assert k.dtype == torch.bfloat16 and k.device == engine.model.device
    assert engine.pool.ssm_state(0, 0).dtype == torch.float32  # mamba_ssm_dtype default


def test_model_type_is_read_from_the_top_level_or_the_text_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"model_type": "qwen3"}}))
    assert read_model_type(tmp_path) == "qwen3"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert read_model_type(tmp_path) == "qwen3_5"


def test_unknown_model_type_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="unsupported model_type"):
        load_model(tmp_path, torch.float32, "cpu", KernelConfig())

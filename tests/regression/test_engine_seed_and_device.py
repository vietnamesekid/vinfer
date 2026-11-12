"""``EngineConfig.seed`` must seed the requests that do not give one, and ``model.device`` must
follow the weights.

Bug: the seed was never used (unseeded requests drew from Python's global RNG, so a server run
was not reproducible), and the model kept its own copy of the device, which went stale after
``model.to(...)``.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from tests.unit.models import fixtures
from vinfer.runtime import SamplingParams
from vinfer.scheduler.serial import SerialScheduler

pytest.importorskip("transformers")


def _tokens(model, seed: int) -> list[list[int]]:
    scheduler = SerialScheduler(model, model.new_state_pool(1, 64), seed=seed)
    params = SamplingParams(max_tokens=8, temperature=1.0, ignore_eos=True)  # no request seed

    async def run():
        return [[s.token_id async for s in scheduler.stream([3, 9, 27], params)] for _ in range(3)]

    try:
        return asyncio.run(run())
    finally:
        scheduler.close()


def test_engine_seed_makes_unseeded_requests_reproducible(tmp_path):
    model = fixtures.load(fixtures.save(fixtures.hf_qwen3(), tmp_path))
    first = _tokens(model, seed=5)
    assert first == _tokens(model, seed=5)
    assert first != _tokens(model, seed=6)
    assert first[0] != first[1]  # requests in one run still get different seeds


def test_model_device_follows_the_weights(tmp_path):
    model = fixtures.load(fixtures.save(fixtures.hf_qwen3(), tmp_path))
    assert model.device == torch.device("cpu")
    model.to("meta")
    assert model.device.type == "meta"

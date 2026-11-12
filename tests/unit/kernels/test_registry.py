"""Backend registries: resolution rules, and the model reaching kernels only through them."""

from __future__ import annotations

import pytest
import torch

from tests.unit.models import fixtures
from vinfer.config import KernelConfig
from vinfer.kernels import resolve_kernels
from vinfer.kernels.attention import ATTENTION, ReferenceAttention
from vinfer.kernels.gdn import GDN, ReferenceGDN
from vinfer.kernels.registry import BackendRegistry
from vinfer.models.qwen import load_qwen

CPU, CUDA = torch.device("cpu"), torch.device("cuda")


def _registry():
    reg: BackendRegistry[object] = BackendRegistry("demo")

    @reg.register("anywhere")
    class Anywhere:
        pass

    @reg.register("cuda-bf16", devices={"cuda"}, dtypes={torch.bfloat16})
    class CudaOnly:
        pass

    return reg, Anywhere, CudaOnly


def test_resolve_returns_a_new_instance_of_the_named_backend():
    reg, anywhere, cuda_only = _registry()
    assert isinstance(reg.resolve("anywhere", CPU, torch.float32), anywhere)
    assert isinstance(reg.resolve("cuda-bf16", CUDA, torch.bfloat16), cuda_only)


def test_unsupported_device_dtype_or_name_fails_instead_of_falling_back():
    reg, _, _ = _registry()
    with pytest.raises(ValueError, match="does not run on cpu"):
        reg.resolve("cuda-bf16", CPU, torch.bfloat16)
    with pytest.raises(ValueError, match=r"does not support torch\.float32"):
        reg.resolve("cuda-bf16", CUDA, torch.float32)
    with pytest.raises(ValueError, match="available: \\['anywhere', 'cuda-bf16'\\]"):
        reg.resolve("missing", CPU, torch.float32)


def test_a_name_registers_once():
    reg, _, _ = _registry()
    with pytest.raises(ValueError, match="twice"):
        reg.register("anywhere")(object)


def test_default_kernel_config_resolves_to_the_references():
    kernels = resolve_kernels(KernelConfig(), CPU, torch.float32)
    assert isinstance(kernels.attention, ReferenceAttention)
    assert isinstance(kernels.gdn, ReferenceGDN)


# --- the model goes through the registry ------------------------------------------------------

CALLS: dict[str, int] = {"attention": 0, "conv": 0, "rule": 0}


class SpyAttention(ReferenceAttention):
    def forward(self, *args, **kwargs):
        CALLS["attention"] += 1
        return super().forward(*args, **kwargs)


class SpyGDN(ReferenceGDN):
    def conv(self, *args, **kwargs):
        CALLS["conv"] += 1
        return super().conv(*args, **kwargs)

    def rule(self, *args, **kwargs):
        CALLS["rule"] += 1
        return super().rule(*args, **kwargs)


if "spy-test" not in ATTENTION.names():
    ATTENTION.register("spy-test")(SpyAttention)
    GDN.register("spy-test")(SpyGDN)


def test_model_calls_the_configured_backends(tmp_path):
    pytest.importorskip("transformers")
    path = fixtures.save(fixtures.hf_qwen3_5(), tmp_path)
    reference = load_qwen(path, torch.float32, "cpu")
    spied = load_qwen(
        path,
        torch.float32,
        "cpu",
        KernelConfig(attention_backend="spy-test", gdn_backend="spy-test"),
    )
    ids = torch.arange(1, 20)
    outs = []
    for model in (reference, spied):
        pool = model.new_state_pool(1, 32)
        outs.append(model.logits(model.forward_hidden(ids, 0, pool.alloc(), pool)))
    for k in CALLS:
        CALLS[k] = 0
    pool = spied.new_state_pool(1, 32)
    spied.forward_hidden(ids, 0, pool.alloc(), pool)
    layer_types = spied.cfg.layer_types
    assert CALLS["attention"] == layer_types.count("full_attention")
    assert CALLS["conv"] == CALLS["rule"] == layer_types.count("linear_attention")
    assert torch.equal(outs[0], outs[1])

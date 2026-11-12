"""Backend registries: resolution rules, and the model reaching kernels only through them."""

from __future__ import annotations

import pytest
import torch

from tests.unit.models import fixtures
from vinfer.config import KernelConfig
from vinfer.kernels import resolve_kernels
from vinfer.kernels.attention import ATTENTION, ReferenceAttention
from vinfer.kernels.gdn import GDN, ReferenceGDN
from vinfer.kernels.registry import BackendContext, BackendRegistry

CPU, CUDA = torch.device("cpu"), torch.device("cuda")


def _ctx(device, dtype=torch.float32):
    return BackendContext(device, dtype)


def _registry():
    reg: BackendRegistry[object] = BackendRegistry("demo")

    @reg.register("anywhere")
    class Anywhere:
        def __init__(self, ctx):
            self.ctx = ctx

    @reg.register("cuda-bf16", devices={"cuda"}, dtypes={torch.bfloat16})
    class CudaOnly:
        def __init__(self, ctx):
            self.ctx = ctx

    @reg.register("needs-sm90", devices={"cuda"}, supports=lambda ctx: "needs compute 9.0")
    class Picky:
        def __init__(self, ctx):
            self.ctx = ctx

    return reg, Anywhere, CudaOnly


def test_resolve_builds_the_backend_with_its_context():
    reg, anywhere, cuda_only = _registry()
    backend = reg.resolve("anywhere", _ctx(CPU))
    assert isinstance(backend, anywhere) and backend.ctx == _ctx(CPU)
    assert isinstance(reg.resolve("cuda-bf16", _ctx(CUDA, torch.bfloat16)), cuda_only)


def test_unsupported_context_or_name_fails_instead_of_falling_back():
    reg, _, _ = _registry()
    with pytest.raises(ValueError, match="does not run on cpu"):
        reg.resolve("cuda-bf16", _ctx(CPU, torch.bfloat16))
    with pytest.raises(ValueError, match=r"does not support torch\.float32"):
        reg.resolve("cuda-bf16", _ctx(CUDA))
    with pytest.raises(ValueError, match=r"needs compute 9\.0"):
        reg.resolve("needs-sm90", _ctx(CUDA))
    with pytest.raises(ValueError, match="available: \\['anywhere', 'cuda-bf16', 'needs-sm90'\\]"):
        reg.resolve("missing", _ctx(CPU))
    assert reg.unsupported("anywhere", _ctx(CPU)) is None


def test_a_name_registers_once():
    reg, _, _ = _registry()
    with pytest.raises(ValueError, match="twice"):
        reg.register("anywhere")(object)


def test_default_kernel_config_resolves_to_the_references():
    kernels = resolve_kernels(KernelConfig(), CPU, torch.float32)
    assert isinstance(kernels.attention, ReferenceAttention)
    assert isinstance(kernels.gdn, ReferenceGDN)


def test_load_model_resolves_kernels_and_fails_fast(tmp_path):
    pytest.importorskip("transformers")
    path = fixtures.save(fixtures.hf_qwen3(), tmp_path)
    with pytest.raises(ValueError, match="unknown attention backend 'missing'"):
        fixtures.load(path, kernels=KernelConfig(attention_backend="missing"))


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
    reference = fixtures.load(path)
    spied = fixtures.load(
        path, kernels=KernelConfig(attention_backend="spy-test", gdn_backend="spy-test")
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

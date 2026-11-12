import torch

from vinfer.config import EngineConfig


def test_defaults_use_reference_paths():
    cfg = EngineConfig(model="dummy")
    assert cfg.kernels.attention_backend == "reference"
    assert cfg.kernels.gdn_backend == "reference"


def test_engine_config_is_the_source_of_device_and_dtype():
    cfg = EngineConfig(model="dummy", dtype="float32", device="cpu")
    assert cfg.torch_dtype == torch.float32 and cfg.torch_device == torch.device("cpu")


def test_serving_parsers_default_to_the_model_family():
    cfg = EngineConfig(model="dummy")
    assert (cfg.serving.reasoning_parser, cfg.serving.tool_call_parser) == ("auto", "auto")


def test_kernel_config_from_string():
    import pytest

    from vinfer.config import KernelConfig

    assert KernelConfig.from_string(None) == KernelConfig() == KernelConfig.from_string("")
    cfg = KernelConfig.from_string(" attention_backend = reference ,gdn_backend=reference")
    assert cfg.describe() == "attention_backend=reference,gdn_backend=reference"
    with pytest.raises(ValueError, match="unknown kernel setting"):
        KernelConfig.from_string("attn=x")
    with pytest.raises(ValueError, match="key=value"):
        KernelConfig.from_string("reference")

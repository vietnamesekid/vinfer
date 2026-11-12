import torch

from vinfer.config import EngineConfig


def test_defaults_use_reference_paths():
    cfg = EngineConfig(model="dummy")
    assert cfg.kernels.attention_backend == "reference"
    assert cfg.kernels.gdn_backend == "reference"


def test_engine_config_is_the_source_of_device_and_dtype():
    cfg = EngineConfig(model="dummy", dtype="float32", device="cpu")
    assert cfg.torch_dtype == torch.float32 and cfg.torch_device == torch.device("cpu")

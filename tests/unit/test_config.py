from vinfer.config import EngineConfig


def test_defaults_use_reference_paths():
    cfg = EngineConfig(model="dummy")
    assert cfg.kernels.attention_backend == "reference"

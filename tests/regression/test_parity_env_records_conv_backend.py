"""The parity environment must record every optional GDN kernel library HF picks up.

Bug: ``environment()`` recorded only whether flash-linear-attention was importable. With
``causal_conv1d`` installed, HF switches the Gated DeltaNet short convolution to a CUDA kernel,
which changes the logits, yet the parity reference cache key and the noise-floor file name
stayed the same, so a reference computed with one conv kernel could be reused for the other,
and the "fla" and "fla + causal_conv1d" noise-floor runs wrote the same file.
"""

from __future__ import annotations

from tests.parity import candidates, noise_floor


def _fake_installed(monkeypatch, present: set[str]) -> None:
    monkeypatch.setattr(candidates, "_installed", lambda module: module in present)


def test_environment_differs_with_causal_conv1d(monkeypatch):
    _fake_installed(monkeypatch, {"fla"})
    fla_only = candidates.environment()
    _fake_installed(monkeypatch, {"fla", "causal_conv1d"})
    fla_cconv = candidates.environment()
    assert fla_only["conv_backend"] == "torch"
    assert fla_cconv["conv_backend"] == "causal_conv1d"
    assert fla_only != fla_cconv


def test_noise_floor_file_name_differs_with_causal_conv1d(monkeypatch):
    names = {}
    for present in [set(), {"fla"}, {"fla", "causal_conv1d"}]:
        _fake_installed(monkeypatch, present)
        names[frozenset(present)] = noise_floor.default_out().name
    assert names == {
        frozenset(): "noise-floor-torch.json",
        frozenset({"fla"}): "noise-floor-fla.json",
        frozenset({"fla", "causal_conv1d"}): "noise-floor-fla-cconv.json",
    }

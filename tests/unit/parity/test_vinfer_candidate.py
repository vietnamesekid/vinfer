"""The vinfer side of the parity tools, on tiny models (CPU): the same code paths the GPU parity
tests use, checked where the answer is known (fp32, same weights as HF)."""

from __future__ import annotations

import pytest

from tests.parity.candidates import HFCandidate, HFConfig, VinferCandidate
from tests.parity.layer_diff import layer_diffs
from tests.parity.logits import compare
from tests.unit.models import fixtures

pytest.importorskip("transformers")


@pytest.fixture(scope="module", params=["qwen3", "qwen3_5"])
def pair(request, tmp_path_factory):
    hf = {"qwen3": fixtures.hf_qwen3, "qwen3_5": fixtures.hf_qwen3_5}[request.param]()
    ours = fixtures.load(fixtures.save(hf, tmp_path_factory.mktemp(request.param)))
    return hf, ours


PROMPTS = [[1, 5, 9, 33, 70, 2], list(range(3, 90)), [7] * 40]


def test_vinfer_candidate_matches_hf_candidate(pair):
    hf_model, ours = pair
    hf = HFCandidate(HFConfig("tiny", dtype="float32", device="cpu", chunk=13), model=hf_model)
    # prefill_chunk 16 splits the 87-token prompt into 6 prefill calls, as long prompts on GPU.
    vinfer = VinferCandidate("", model=ours, prefill_chunk=16, lm_head_chunk=13)
    for ref, got in zip(hf.summaries(PROMPTS), vinfer.summaries(PROMPTS), strict=True):
        metrics = compare(ref, got)
        assert metrics["argmax_mismatch"] == 0 and metrics["max_abs_logit"] < 1e-3, metrics


def test_layer_diffs_are_tiny_when_the_models_agree(pair):
    hf_model, ours = pair
    rows = layer_diffs(hf_model, ours, PROMPTS[1])
    assert [r["layer"] for r in rows] == [*range(ours.cfg.num_hidden_layers), "logits"]
    assert all(r["max_rel"] < 1e-4 for r in rows), rows


def test_layer_diffs_point_at_the_broken_layer(pair):
    hf_model, ours = pair
    layer = ours.layers[1]
    weight = layer.mlp.down_proj.weight
    saved = weight.clone()
    try:
        weight.mul_(1.5)  # corrupt layer 1 only
        rows = layer_diffs(hf_model, ours, PROMPTS[1])
    finally:
        weight.copy_(saved)
    assert rows[0]["max_rel"] < 1e-4 and rows[1]["max_rel"] > 1e-2

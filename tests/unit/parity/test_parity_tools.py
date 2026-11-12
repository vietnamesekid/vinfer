"""Parity tooling on CPU with a tiny random Qwen3.5 (GDN + attention layers)."""

from __future__ import annotations

import pytest
import torch

from tests.parity import prompt_set
from tests.parity.candidates import HFCandidate, HFConfig
from tests.parity.logits import LogitSummary, aggregate, compare, full_positions, summarize

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=256,
    )
    assert cfg.layer_types.count("linear_attention") == 3  # hybrid, like the real model
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg).float().eval()


def _cand(model, **kw):
    return HFCandidate(HFConfig("tiny", dtype="float32", device="cpu", chunk=7, **kw), model=model)


PROMPTS = [list(range(1, 40)), list(range(5, 18)), [3, 1, 4, 1, 5, 9, 2, 6]]


def test_summary_matches_full_logits(tiny_model):
    s = _cand(tiny_model).summaries(PROMPTS[:1])[0]
    with torch.no_grad():
        full = tiny_model(torch.tensor([PROMPTS[0]])).logits[0].float()
    assert torch.equal(s.argmax, full.argmax(-1))
    torch.testing.assert_close(s.lse, torch.logsumexp(full, -1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s.full, full[s.full_pos], rtol=1e-5, atol=1e-5)


def test_self_compare_is_exact(tiny_model, tmp_path):
    a = _cand(tiny_model).summaries(PROMPTS)
    b = _cand(tiny_model).summaries(PROMPTS)
    agg = aggregate({str(i): compare(x, y) for i, (x, y) in enumerate(zip(a, b, strict=True))})
    assert agg["bitwise_equal"] and agg["argmax_mismatch"] == 0 and agg["kl_max"] == 0
    a[0].save(tmp_path / "x.safetensors")
    assert compare(a[0], LogitSummary.load(tmp_path / "x.safetensors"))["bitwise_equal"]


@pytest.mark.parametrize("pad", ["right", "left"])
def test_batched_and_shuffled_close_to_single(tiny_model, pad):
    ref = _cand(tiny_model).summaries(PROMPTS)
    got = _cand(tiny_model, batch_size=3, padding_side=pad, order_seed=1).summaries(PROMPTS)
    for r, g in zip(ref, got, strict=True):
        m = compare(r, g)
        assert m["argmax_mismatch"] == 0 and m["max_abs_logit"] < 1e-3, (pad, m)


def test_compare_detects_perturbation():
    torch.manual_seed(0)
    logits = torch.randn(10, 50)
    pos = full_positions(10, n_last=3, n_random=2)
    ref = summarize([logits[:4], logits[4:]], pos, k=5)
    bumped = logits.clone()
    bumped[2, ref.argmax[2]] -= 100  # flips the argmax at position 2
    m = compare(ref, summarize([bumped], pos, k=5))
    assert m["argmax_mismatch"] == 1 and m["first_mismatch_pos"] == 2


def test_prompt_set_v1_loads_and_tokenizes():
    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [len(w) for w in text.split()]

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [len(m["content"]) for m in messages] + [0]

    prompts = prompt_set.load("v1")
    assert len(prompts) == 50 and len({p["id"] for p in prompts}) == 50
    kinds = {p["kind"] for p in prompts}
    assert {"short", "multilingual", "code", "chat", "raw", "long"} <= kinds
    ids = prompt_set.tokenize(prompts, Tok())
    assert all(ids.values())


def test_noise_floor_runs_end_to_end(tiny_model, tmp_path, monkeypatch):
    import dataclasses
    import json

    from tests.parity import candidates, noise_floor

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [hash(w) % 120 + 1 for w in text.split()][:40]

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [hash(m["content"]) % 120 + 1 for m in messages] + [7, 8]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tok())

    def fake(cfg):
        # Eager vs sdpa and dtype are real knobs on GPU; on CPU keep fp32 for speed.
        cfg = dataclasses.replace(cfg, device="cpu", dtype="float32")
        return candidates.HFCandidate(cfg, model=tiny_model)

    monkeypatch.setattr(noise_floor, "HFCandidate", fake)
    out = tmp_path / "r.json"
    res = noise_floor.run("tiny", None, out, ["bfloat16"])
    assert json.loads(out.read_text())["variants"].keys() == res["variants"].keys()
    assert res["variants"]["bfloat16-vs-fp32"]["aggregate"]["prompts"] == 50
    groups = res["variants"]["bfloat16-vs-fp32"]["groups"]
    assert groups["short"]["prompts"] == 46 and groups["long"]["prompts"] == 4
    assert res["variants"]["bfloat16:repeat"]["aggregate"]["bitwise_equal"]


def test_reference_cache_is_keyed_on_the_software_and_gpu(monkeypatch, tmp_path):
    """A reference computed with another GPU or library version is never reused."""
    from tests.parity import test_parity

    env = {"model": "Qwen/X", "revision": "abc", "dtype": "bfloat16", "cache": tmp_path}
    base = {"torch": "1", "cuda": "12", "transformers": "5", "gpu": "H100", "gdn_backend": "fla"}
    monkeypatch.setattr(test_parity, "environment", lambda: base)
    before = test_parity.reference_dir(env)
    monkeypatch.setattr(test_parity, "environment", lambda: {**base, "transformers": "6"})
    assert test_parity.reference_dir(env) != before

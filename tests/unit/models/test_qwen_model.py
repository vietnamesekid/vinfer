"""The vinfer Qwen models against HF Transformers on tiny random checkpoints, fp32, CPU.

Same weights on both sides (written by HF, read by the vinfer loader), so the logits must agree
to fp32 rounding: this is parity on the formulas, at a small size.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.unit.models import fixtures
from vinfer.models.qwen import QwenConfig, load_qwen

pytest.importorskip("transformers")

TOL = {"atol": 1e-4, "rtol": 1e-4}
BUILDERS = {"qwen3": fixtures.hf_qwen3, "qwen3_5": fixtures.hf_qwen3_5}


def _ids(n: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 96, (n,), generator=torch.Generator().manual_seed(seed))


def _hf_logits(hf, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return hf(ids[None]).logits[0]


@pytest.fixture(scope="module", params=list(BUILDERS))
def pair(request, tmp_path_factory):
    hf = BUILDERS[request.param]()
    ours = load_qwen(
        fixtures.save(hf, tmp_path_factory.mktemp(request.param)), torch.float32, "cpu"
    )
    return request.param, hf, ours


def test_prefill_logits_match_hf(pair):
    _, hf, ours = pair
    ids = _ids(70)  # longer than one 64-token DeltaNet chunk
    pool = ours.new_state_pool(num_slots=1, max_seq_len=128)
    slot = pool.alloc()
    logits = ours.logits(ours.forward_hidden(ids, 0, slot, pool))
    torch.testing.assert_close(logits, _hf_logits(hf, ids), **TOL)


def test_prefill_then_decode_matches_hf_full_forward(pair):
    _, hf, ours = pair
    ids = _ids(20, seed=1)
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    slot = pool.alloc()
    rows = [ours.logits(ours.forward_hidden(ids[:12], 0, slot, pool))]
    for t in range(12, 20):  # one token at a time: recurrent DeltaNet form + KV cache
        rows.append(ours.logits(ours.forward_hidden(ids[t : t + 1], t, slot, pool)))
    torch.testing.assert_close(torch.cat(rows), _hf_logits(hf, ids), **TOL)


def test_prefill_in_two_calls_matches_one(pair):
    _, hf, ours = pair
    ids = _ids(40, seed=2)
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    slot = pool.alloc()
    first = ours.logits(ours.forward_hidden(ids[:17], 0, slot, pool))
    second = ours.logits(ours.forward_hidden(ids[17:], 17, slot, pool))
    torch.testing.assert_close(torch.cat([first, second]), _hf_logits(hf, ids), **TOL)


def test_a_reused_slot_starts_clean(pair):
    # B on a slot that A used gives the same logits as B on a fresh pool.
    _, _, ours = pair
    a, b = _ids(30, seed=3), _ids(25, seed=4)
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    slot = pool.alloc()
    ours.forward_hidden(a, 0, slot, pool)
    pool.free(slot)
    slot = pool.alloc()
    reused = ours.logits(ours.forward_hidden(b, 0, slot, pool))
    fresh_pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    fresh = ours.logits(ours.forward_hidden(b, 0, fresh_pool.alloc(), fresh_pool))
    assert torch.equal(reused, fresh)


def test_multimodal_checkpoint_layout_loads_and_matches(tmp_path):
    hf = fixtures.hf_qwen3_5(seed=5)
    ours = load_qwen(fixtures.save_as_multimodal_checkpoint(hf, tmp_path), torch.float32, "cpu")
    ids = _ids(33, seed=5)
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    logits = ours.logits(ours.forward_hidden(ids, 0, pool.alloc(), pool))
    torch.testing.assert_close(logits, _hf_logits(hf, ids), **TOL)


def test_fp32_tensors_of_a_bf16_checkpoint_stay_fp32(tmp_path):
    hf = fixtures.hf_qwen3_5(seed=6)
    path = fixtures.save_as_multimodal_checkpoint(hf, tmp_path)
    tensors = load_file(str(path / "model.safetensors"))
    keep = ("linear_attn.A_log", "linear_attn.norm.weight")
    tensors = {k: v if k.endswith(keep) else v.bfloat16() for k, v in tensors.items()}
    save_file(tensors, str(path / "model.safetensors"))
    ours = load_qwen(path, torch.bfloat16, "cpu")
    layer = ours.layers[0].linear_attn
    assert layer.A_log.dtype == torch.float32 and layer.norm.weight.dtype == torch.float32
    assert (
        layer.dt_bias.dtype == torch.bfloat16 and ours.embed_tokens.weight.dtype == torch.bfloat16
    )


def test_tied_lm_head_in_checkpoint_must_equal_the_embedding(tmp_path):
    hf = fixtures.hf_qwen3(seed=7)
    path = fixtures.save(hf, tmp_path)
    tensors = load_file(str(path / "model.safetensors"))
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, str(path / "model.safetensors"))
    load_qwen(path, torch.float32, "cpu")  # equal: fine (Qwen3-0.6B stores both)
    tensors["lm_head.weight"][0, 0] += 1
    save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="lm_head"):
        load_qwen(path, torch.float32, "cpu")


def test_missing_and_misshaped_tensors_are_errors(tmp_path):
    hf = fixtures.hf_qwen3(seed=8)
    path = fixtures.save(hf, tmp_path)
    tensors = load_file(str(path / "model.safetensors"))
    name = "model.layers.0.self_attn.q_norm.weight"
    save_file({k: v for k, v in tensors.items() if k != name}, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="not in the checkpoint"):
        load_qwen(path, torch.float32, "cpu")
    tensors[name] = torch.zeros(3)
    save_file(tensors, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="shape"):
        load_qwen(path, torch.float32, "cpu")


def test_kv_capacity_is_enforced(pair):
    _, _, ours = pair
    if "full_attention" not in ours.cfg.layer_types:
        pytest.skip("no attention layer")
    pool = ours.new_state_pool(num_slots=1, max_seq_len=8)
    with pytest.raises(ValueError, match="KV capacity"):
        ours.forward_hidden(_ids(9), 0, pool.alloc(), pool)


def test_config_reads_the_real_qwen3_5_2b_fields():
    raw = {
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "attn_output_gate": True,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
        },
    }
    cfg = QwenConfig.from_hf(json.loads(json.dumps(raw)))
    assert cfg.tie_word_embeddings and cfg.rotary_dim == 64 and cfg.zero_centered_norm
    assert cfg.linear_conv_dim == 6144 and cfg.ssm_dtype == torch.float32
    with pytest.raises(ValueError, match="RoPE"):
        raw["text_config"]["rope_parameters"]["rope_type"] = "yarn"
        QwenConfig.from_hf(raw)

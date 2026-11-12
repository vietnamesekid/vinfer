"""Tiny Qwen3 / Qwen3.5 checkpoints with random weights, built by HF Transformers, for CPU tests.

The Qwen3.5 config has more DeltaNet value heads than key heads and K != V, which the real 2B
checkpoint does not: a wrong key-head mapping or a transposed state fails here instead of passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

QWEN3_KW = {
    "vocab_size": 96,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,  # q_proj wider than hidden, as in Qwen3-0.6B
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}

QWEN3_5_KW = {
    "vocab_size": 96,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 5,
    "layer_types": [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 6,  # HV = 3 H
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 12,  # K != V
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 256,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [1, 1, 0],
    },
    "tie_word_embeddings": True,
}


def _perturb(model: torch.nn.Module, seed: int) -> None:
    # HF initializes zero-centered norms to 0 and gated norms to 1; random values make a wrong
    # norm variant visible.
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * 0.05)


def hf_qwen3(seed: int = 0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**QWEN3_KW)).float().eval()
    _perturb(model, seed)
    return model


def hf_qwen3_5(seed: int = 0):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**QWEN3_5_KW)).float().eval()
    _perturb(model, seed)
    return model


def save(model, path: Path) -> Path:
    """``save_pretrained``: flat config, ``model.*`` tensor names, no ``lm_head`` when tied."""
    model.save_pretrained(path)
    return path


def save_as_multimodal_checkpoint(model, path: Path) -> Path:
    """Lay out a Qwen3.5 text model like the real multimodal checkpoint: config nested under
    ``text_config``, ``model.language_model.*`` names, plus vision and MTP tensors to skip."""
    path.mkdir(parents=True, exist_ok=True)
    text = model.config.to_dict()
    raw = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {**text, "mamba_ssm_dtype": "float32"},
    }
    raw["text_config"].pop("tie_word_embeddings", None)
    (path / "config.json").write_text(json.dumps(raw))
    tensors = {
        "model.language_model." + k.removeprefix("model."): v.detach().clone().contiguous()
        for k, v in model.state_dict().items()
        if k != "lm_head.weight"
    }
    tensors["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 3)
    tensors["mtp.fc.weight"] = torch.zeros(32, 64)
    save_file(tensors, str(path / "model.safetensors"))
    return path


CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant{% endif %}"
)


def save_tiny_tokenizer(path: Path, vocab_size: int = 96) -> Path:
    """A word-level tokenizer with ``vocab_size`` entries (``<|im_end|>`` = eos = id 2) and a
    chat template, saved next to a tiny checkpoint so the whole directory serves like a real
    model."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant", "system"]
    words += [f"w{i}" for i in range(vocab_size - len(words))]
    raw = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", eos_token="<|im_end|>", chat_template=CHAT_TEMPLATE
    )
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    tok.save_pretrained(path)
    return path

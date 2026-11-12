"""The tokenizer wrapper on a tiny offline tokenizer with a chat template (CPU)."""

from __future__ import annotations

import json

import pytest

from vinfer.runtime.tokenizer import Tokenizer

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_tokenizer_dir(tmp_path_factory):
    from tokenizers import Tokenizer as RawTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    words = ["<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant", "hello", "world"]
    raw = RawTokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", eos_token="<|im_end|>", chat_template=TEMPLATE
    )
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    path = tmp_path_factory.mktemp("tok")
    tok.save_pretrained(path)
    (path / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": [2, 6]}}))
    return path


def test_chat_ids_match_hf_apply_chat_template(tiny_tokenizer_dir):
    ours = Tokenizer(tiny_tokenizer_dir)
    messages = [{"role": "user", "content": "hello world"}]
    hf = transformers.AutoTokenizer.from_pretrained(str(tiny_tokenizer_dir))
    expected = hf.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    assert ours.chat_ids(messages) == list(expected) == [1, 3, 5, 6, 2, 1, 4]


def test_eos_ids_come_from_tokenizer_and_config(tiny_tokenizer_dir):
    assert Tokenizer(tiny_tokenizer_dir).eos_token_ids == {2, 6}


def test_decode_skips_special_tokens(tiny_tokenizer_dir):
    ours = Tokenizer(tiny_tokenizer_dir)
    assert ours.decode([1, 5, 6, 2]) == "hello world"
    assert ours.encode("hello world") == [5, 6]


def test_decode_can_keep_special_tokens_to_show_turn_boundaries(tiny_tokenizer_dir):
    ours = Tokenizer(tiny_tokenizer_dir)
    assert ours.decode([1, 4], skip_special_tokens=False) == "<|im_start|> assistant"


def test_tools_reach_the_chat_template(tiny_tokenizer_dir):
    hf = transformers.AutoTokenizer.from_pretrained(str(tiny_tokenizer_dir))
    hf.chat_template = "{% for t in tools %}{{ t['function']['name'] }} {% endfor %}"
    hf.save_pretrained(tiny_tokenizer_dir / "tools")
    with_tools = Tokenizer(tiny_tokenizer_dir / "tools")
    tools = [{"type": "function", "function": {"name": "hello"}}]
    assert with_tools.chat_ids([{"role": "user", "content": "x"}], tools) == [5]

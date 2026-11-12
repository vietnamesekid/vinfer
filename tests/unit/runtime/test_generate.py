"""The prefill/decode loop on tiny models (CPU): greedy output equals HF ``generate``; stop
conditions and streamed text."""

from __future__ import annotations

import pytest
import torch

from tests.unit.models import fixtures
from vinfer.models.qwen import load_qwen
from vinfer.runtime import SamplingParams, generate
from vinfer.runtime.generate import _TextStream

pytest.importorskip("transformers")

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def decode(ids: list[int]) -> str:
    return "".join(LETTERS[i % 26] for i in ids)


@pytest.fixture(scope="module", params=["qwen3", "qwen3_5"])
def models(request, tmp_path_factory):
    hf = {"qwen3": fixtures.hf_qwen3, "qwen3_5": fixtures.hf_qwen3_5}[request.param]()
    ours = load_qwen(
        fixtures.save(hf, tmp_path_factory.mktemp(request.param)), torch.float32, "cpu"
    )
    return hf, ours


PROMPT = [5, 17, 3, 88, 42, 9, 61, 70, 12]


def test_greedy_generation_equals_hf_generate(models):
    hf, ours = models
    with torch.no_grad():
        expected = hf.generate(
            torch.tensor([PROMPT]),
            max_new_tokens=15,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )[0, len(PROMPT) :].tolist()
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    steps = list(generate(ours, pool, PROMPT, SamplingParams(max_tokens=15, temperature=0)))
    assert [s.token_id for s in steps] == expected
    assert steps[-1].finish_reason == "length" and pool.num_free == 1  # slot returned


def test_eos_stops_unless_ignored(models):
    _, ours = models
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    greedy = SamplingParams(max_tokens=10, temperature=0)
    tokens = [s.token_id for s in generate(ours, pool, PROMPT, greedy)]
    eos = frozenset({tokens[3]})
    first_eos = tokens.index(tokens[3])
    stopped = list(generate(ours, pool, PROMPT, greedy, eos_token_ids=eos))
    assert len(stopped) == first_eos + 1 and stopped[-1].finish_reason == "stop"
    ignored = SamplingParams(max_tokens=10, temperature=0, ignore_eos=True)
    assert len(list(generate(ours, pool, PROMPT, ignored, eos_token_ids=eos))) == 10


def test_streamed_text_is_the_decoded_output(models):
    _, ours = models
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    steps = list(
        generate(ours, pool, PROMPT, SamplingParams(max_tokens=12, temperature=0), decode=decode)
    )
    assert "".join(s.text for s in steps) == decode([s.token_id for s in steps])


def test_stop_string_ends_the_text_before_it(models):
    _, ours = models
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    greedy = SamplingParams(max_tokens=12, temperature=0)
    full = decode([s.token_id for s in generate(ours, pool, PROMPT, greedy)])
    stop = full[5:7]
    params = SamplingParams(max_tokens=12, temperature=0, stop=(stop,))
    steps = list(generate(ours, pool, PROMPT, params, decode=decode))
    assert "".join(s.text for s in steps) == full[: full.index(stop)]
    assert steps[-1].finish_reason == "stop"


def test_sampling_is_reproducible_with_a_seed(models):
    _, ours = models
    pool = ours.new_state_pool(num_slots=1, max_seq_len=64)
    params = SamplingParams(max_tokens=10, temperature=1.0, seed=3)
    runs = [[s.token_id for s in generate(ours, pool, PROMPT, params)] for _ in range(2)]
    assert runs[0] == runs[1]


def test_prompt_plus_max_tokens_must_fit(models):
    _, ours = models
    # 9 prompt tokens + 4 generated: the last one is never fed back, so 12 positions suffice.
    pool = ours.new_state_pool(num_slots=1, max_seq_len=12)
    params = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)
    assert len(list(generate(ours, pool, PROMPT, params))) == 4
    with pytest.raises(ValueError, match="max_seq_len"):
        next(generate(ours, pool, PROMPT, SamplingParams(max_tokens=5)))


# --- text stream ---------------------------------------------------------------------------------


def test_text_held_back_while_it_could_be_a_stop_string():
    stream = _TextStream(decode, ("cde",))
    assert stream.push(0) == ("", False)  # "a": could start "cde"? no, but held (hold = 2)
    assert stream.push(1) == ("", False)  # "ab"
    assert stream.push(2) == ("a", False)  # "abc": "c" and "b" held
    assert stream.push(3) == ("b", False)  # "abcd"
    assert stream.push(4) == ("", True)  # "abcde": stop found at 2, "ab" already sent


def test_unfinished_utf8_is_held_back():
    pieces = {0: "a", 1: "�", 2: "é"}

    def partial(ids):
        # token 1 alone decodes to a replacement char; with token 2 it becomes "é"
        text = "".join(pieces[i] for i in ids)
        return text.replace("�é", "é")

    stream = _TextStream(partial, ())
    assert stream.push(0) == ("a", False)
    assert stream.push(1) == ("", False)
    assert stream.push(2) == ("é", False)

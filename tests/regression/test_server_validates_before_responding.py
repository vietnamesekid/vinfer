"""The server must reject a request that cannot run with a 400, before any response starts.

Bug: the capacity check ran lazily inside the generator on the worker thread, so a too-long
request got HTTP 500 (non-streaming) or a 200 stream ending in an error event. Also,
``max_tokens=0`` was silently replaced by 16, and an omitted ``max_tokens`` meant 16 tokens
instead of the rest of the context (vLLM's behaviour).
"""

from __future__ import annotations

import pytest

from tests.unit.models import fixtures
from vinfer.runtime.tokenizer import Tokenizer
from vinfer.scheduler.serial import SerialScheduler
from vinfer.server.app import create_app

pytest.importorskip("transformers")
testclient = pytest.importorskip("fastapi.testclient")

MAX_SEQ_LEN = 24
MESSAGES = [{"role": "user", "content": "w1 w2 w3"}]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = fixtures.save(fixtures.hf_qwen3(), tmp_path_factory.mktemp("val"))
    fixtures.save_tiny_tokenizer(path)
    model = fixtures.load(path)
    tok = Tokenizer(path)
    scheduler = SerialScheduler(
        model, model.new_state_pool(1, MAX_SEQ_LEN), tok.eos_token_ids, tok.decode
    )
    with testclient.TestClient(create_app(scheduler, tok, "tiny")) as c:
        yield c, len(tok.chat_ids(MESSAGES))
    scheduler.close()


def _post(c, **body):
    return c.post("/v1/chat/completions", json={"model": "tiny", "messages": MESSAGES, **body})


@pytest.mark.parametrize("stream", [False, True])
def test_too_long_is_400_before_the_stream_starts(client, stream):
    c, prompt_len = client
    r = _post(c, max_tokens=MAX_SEQ_LEN - prompt_len + 2, stream=stream)
    assert r.status_code == 400 and "max_seq_len" in r.text


def test_exactly_full_context_is_accepted(client):
    c, prompt_len = client
    r = _post(c, max_tokens=MAX_SEQ_LEN - prompt_len + 1, ignore_eos=True, temperature=0)
    assert r.status_code == 200
    assert r.json()["usage"]["completion_tokens"] == MAX_SEQ_LEN - prompt_len + 1


def test_zero_max_tokens_is_rejected_not_replaced(client):
    c, _ = client
    assert _post(c, max_tokens=0).status_code == 400
    assert _post(c, max_completion_tokens=0, max_tokens=5).status_code == 400


def test_omitted_max_tokens_means_the_rest_of_the_context(client):
    c, prompt_len = client
    r = _post(c, ignore_eos=True, temperature=0)
    assert r.json()["usage"]["completion_tokens"] == MAX_SEQ_LEN - prompt_len + 1


@pytest.mark.parametrize(
    "bad", [{"stop": [1]}, {"stop": 5}, {"stop_token_ids": ["a"]}, {"stop_token_ids": [True]}]
)
def test_malformed_stop_settings_are_400_not_an_error_inside_the_stream(client, bad):
    # Bug: a non-string stop entry passed validation and failed in the generation thread,
    # inside a 200 stream.
    c, _ = client
    assert _post(c, max_tokens=3, stream=True, **bad).status_code == 400


def test_a_template_rejecting_the_messages_is_400(tmp_path):
    # Bug: Qwen templates call raise_exception (a jinja2 TemplateError), which the server did
    # not catch, so the client got a 500.
    from transformers import AutoTokenizer

    path = fixtures.save(fixtures.hf_qwen3(), tmp_path)
    fixtures.save_tiny_tokenizer(path)
    tok = AutoTokenizer.from_pretrained(str(path))
    tok.chat_template = (
        "{% if messages[-1]['role'] != 'user' %}{{ raise_exception('last turn must be user') }}"
        "{% endif %}" + fixtures.CHAT_TEMPLATE
    )
    tok.save_pretrained(path)
    model = fixtures.load(path)
    tokenizer = Tokenizer(path)
    scheduler = SerialScheduler(
        model, model.new_state_pool(1, MAX_SEQ_LEN), tokenizer.eos_token_ids
    )
    with testclient.TestClient(create_app(scheduler, tokenizer, "tiny")) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "tiny", "messages": [{"role": "system", "content": "w1"}]},
        )
    scheduler.close()
    assert r.status_code == 400 and "last turn must be user" in r.text

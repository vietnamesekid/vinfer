"""The chat API must follow OpenAI's request and stream format.

Bugs:
1. Errors were FastAPI's ``{"detail": ...}``; OpenAI clients read ``{"error": {"message",
   "type", "param", "code"}}``.
2. Parameters the server does not implement (``logprobs``, penalties, ``response_format``, ...)
   were silently ignored, so a client got output that did not do what it asked.
3. ``content`` given as an array of parts reached the chat template raw; text parts must be
   joined, other parts rejected.
5. With ``stream_options.include_usage`` the chunks before the usage chunk had no ``usage``
   field; OpenAI sends ``"usage": null`` on each of them.
"""

from __future__ import annotations

import json

import pytest

from tests.unit.server.fakes import FakeScheduler, FakeTokenizer
from vinfer.server.app import create_app

testclient = pytest.importorskip("fastapi.testclient")

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def api():
    tokenizer = FakeTokenizer("")
    client = testclient.TestClient(create_app(FakeScheduler(["a", "b"]), tokenizer, "tiny"))
    return client, tokenizer


def _post(client, **body):
    return client.post("/v1/chat/completions", json={"model": "tiny", **body})


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({}, "messages"),  # missing
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "robot", "content": "x"}]}, "messages.0.role"),
        ({"messages": MESSAGES, "temperature": -1}, None),  # rejected by the sampler
        ({"messages": MESSAGES, "stop": [1]}, "stop"),
        ({"messages": MESSAGES, "max_tokens": 0}, None),
    ],
)
def test_errors_are_openai_shaped(api, body, param):
    r = _post(api[0], **body)
    assert r.status_code == 400
    assert set(r.json()) == {"error"}
    err = r.json()["error"]
    assert set(err) == {"message", "type", "param", "code"}
    assert err["type"] == "BadRequestError" and err["code"] == 400 and err["message"]
    if param is not None:
        assert err["param"] == param or err["param"].startswith(param)


def test_a_template_rejecting_the_messages_is_an_openai_400(api):
    r = _post(api[0], messages=MESSAGES, chat_template_kwargs={"fail": True})
    assert r.status_code == 400 and "rejected" in r.json()["error"]["message"]


@pytest.mark.parametrize(
    "extra",
    [
        {"logprobs": True},
        {"top_logprobs": 2},
        {"presence_penalty": 0.5},
        {"frequency_penalty": -1},
        {"repetition_penalty": 1.1},
        {"logit_bias": {"5": 10}},
        {"response_format": {"type": "json_object"}},
        {"n": 2},
        {"reasoning_effort": "low"},
        {"functions": [{"name": "f"}]},
    ],
)
def test_unsupported_parameters_are_rejected_not_ignored(api, extra):
    r = _post(api[0], messages=MESSAGES, **extra)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == next(iter(extra))


def test_unsupported_parameters_at_their_default_are_accepted(api):
    defaults = {
        "logprobs": False,
        "presence_penalty": 0,
        "frequency_penalty": 0.0,
        "logit_bias": {},
        "response_format": {"type": "text"},
        "n": 1,
        "user": "someone",
        "some_vendor_extra": 1,
    }
    assert _post(api[0], messages=MESSAGES, **defaults).status_code == 200


def test_text_parts_are_joined_for_the_template(api):
    client, tokenizer = api
    parts = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
    assert _post(client, messages=[{"role": "user", "content": parts}]).status_code == 200
    assert tokenizer.calls[-1]["messages"] == [{"role": "user", "content": "first\nsecond"}]


def test_non_text_parts_are_400(api):
    parts = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
    ]
    r = _post(api[0], messages=[{"role": "user", "content": parts}])
    assert r.status_code == 400 and "image_url" in r.json()["error"]["message"]


def test_every_chunk_before_the_usage_chunk_has_usage_null(api):
    r = _post(api[0], messages=MESSAGES, stream=True, stream_options={"include_usage": True})
    lines = [ln.removeprefix("data: ") for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    events = [json.loads(ln) for ln in lines[:-1]]
    assert all("usage" in e and e["usage"] is None for e in events[:-1])
    assert events[-1]["choices"] == [] and events[-1]["usage"]["completion_tokens"] == 2

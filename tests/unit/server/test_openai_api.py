"""The chat API's request and response format, on a scheduler that "generates" known text."""

from __future__ import annotations

import json

import pytest

from tests.unit.server.fakes import FakeScheduler, FakeTokenizer
from vinfer.server.app import create_app
from vinfer.server.parsers import OutputParsers

testclient = pytest.importorskip("fastapi.testclient")

THINKING = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]
# Reasoning, a sentence, then a call, cut into tokens that split every tag.
QWEN35_OUTPUT = [
    "Need", " weather.", "\n</th", "ink>\n\n", "Checking", ".\n\n<tool", "_call>\n<func",
    "tion=get_weather>\n<parameter=city>\nHu", "ế\n</parameter>\n<parameter=days>\n2",
    "\n</parameter>\n</function>\n</tool_call>",
]  # fmt: skip


QWEN_PARSERS = OutputParsers("qwen3", "qwen3_coder")


def _client(pieces=None, prompt=THINKING, parsers=QWEN_PARSERS):
    scheduler, tokenizer = FakeScheduler(pieces), FakeTokenizer(prompt)
    client = testclient.TestClient(
        create_app(scheduler, tokenizer, "tiny", parsers), raise_server_exceptions=False
    )
    return client, scheduler, tokenizer


def _events(response) -> list[dict]:
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    return [json.loads(ln.removeprefix("data: ")) for ln in lines[:-1]]


def _post(client, **body):
    return client.post("/v1/chat/completions", json={"model": "tiny", "messages": MESSAGES, **body})


def test_non_streaming_reasoning_content_and_tool_calls():
    client, _, _ = _client(QWEN35_OUTPUT)
    r = _post(client, tools=TOOLS)
    assert r.status_code == 200
    out = r.json()
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert msg["role"] == "assistant"
    assert msg["reasoning_content"] == "Need weather."
    assert msg["content"] == "Checking."
    [call] = msg["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("chatcmpl-tool-")
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Huế", "days": 2}
    assert out["usage"] == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    assert out["object"] == "chat.completion" and out["model"] == "tiny"


def test_streaming_sends_one_chunk_per_token_and_assembles_the_same_message():
    client, _, _ = _client(QWEN35_OUTPUT)
    events = _events(_post(client, tools=TOOLS, stream=True))
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    tokens = events[1:]
    assert len(tokens) == len(QWEN35_OUTPUT)  # held-back text still gets its chunk
    assert all("usage" not in e for e in events)  # not asked for
    reasoning = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in tokens)
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in tokens)
    calls = [c for e in tokens for c in e["choices"][0]["delta"].get("tool_calls", [])]
    assert (reasoning, content) == ("Need weather.", "Checking.")
    assert calls[0]["index"] == 0 and calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["id"].startswith("chatcmpl-tool-") and calls[0]["type"] == "function"
    assert all("id" not in c and "name" not in c["function"] for c in calls[1:])
    args = "".join(c["function"].get("arguments", "") for c in calls)
    assert json.loads(args) == {"city": "Huế", "days": 2}
    assert [e["choices"][0]["finish_reason"] for e in tokens] == [None] * 9 + ["tool_calls"]
    for e in tokens:  # no token chunk is empty: an empty delta is content ""
        assert e["choices"][0]["delta"]


def test_thinking_off_prompt_gives_plain_content():
    prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    client, _, _ = _client(["4", "."], prompt=prompt)
    msg = _post(client).json()["choices"][0]["message"]
    assert (msg["content"], msg["reasoning_content"], msg["tool_calls"]) == ("4.", None, [])


def test_tool_choice_none_leaves_calls_in_the_content_and_still_shows_the_tools():
    client, _, tokenizer = _client(["<tool_call>x</tool_call>"], prompt="")
    r = _post(client, tools=TOOLS, tool_choice="none")
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "<tool_call>x</tool_call>" and msg["tool_calls"] == []
    assert r.json()["choices"][0]["finish_reason"] == "stop"
    assert tokenizer.calls[-1]["tools"][0]["function"]["name"] == "get_weather"


def test_finish_reason_stays_length_when_cut_inside_a_call():
    client, _, _ = _client(QWEN35_OUTPUT)
    choice = _post(client, tools=TOOLS, max_tokens=8).json()["choices"][0]
    assert choice["finish_reason"] == "length"


def test_assistant_tool_call_arguments_reach_the_template_as_a_dict():
    client, _, tokenizer = _client(["ok"], prompt="")
    history = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Huế"}'},
                },
                {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{bad"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "sunny"}]},
    ]
    assert _post(client, messages=history, tools=TOOLS).status_code == 200
    sent = tokenizer.calls[-1]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Huế"}
    assert sent[1]["tool_calls"][1]["function"]["arguments"] == {}  # malformed JSON, as vLLM
    assert sent[2] == {"role": "tool", "tool_call_id": "c1", "content": "sunny"}
    assert "content" not in sent[1]


def test_chat_template_kwargs_reach_the_template():
    client, _, tokenizer = _client(["ok"], prompt="")
    _post(client, chat_template_kwargs={"enable_thinking": True})
    assert tokenizer.calls[-1]["kwargs"] == {"enable_thinking": True}


def test_continuous_usage_stats_put_usage_on_every_chunk():
    client, _, _ = _client(["a", "b", "c"], prompt="")
    opts = {"include_usage": True, "continuous_usage_stats": True}
    events = _events(_post(client, stream=True, stream_options=opts))
    assert [e["usage"]["completion_tokens"] for e in events] == [0, 1, 2, 3, 3]
    assert events[-1]["choices"] == [] and events[-1]["usage"]["total_tokens"] == 8


def test_continuous_usage_needs_include_usage_as_in_vllm():
    client, _, _ = _client(["a"], prompt="")
    events = _events(_post(client, stream=True, stream_options={"continuous_usage_stats": True}))
    assert all("usage" not in e for e in events)


def test_models_card():
    client, scheduler, _ = _client()
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    [card] = data["data"]
    assert card["id"] == "tiny" and card["object"] == "model" and card["owned_by"] == "vinfer"
    assert isinstance(card["created"], int) and card["max_model_len"] == scheduler.max_seq_len


def test_sampling_fields_reach_the_scheduler():
    client, scheduler, _ = _client(["a"], prompt="")
    body = dict(temperature=0.5, top_p=0.9, top_k=7, seed=3, stop="x", stop_token_ids=[9])
    assert _post(client, max_completion_tokens=4, ignore_eos=True, **body).status_code == 200
    p = scheduler.params
    assert (p.temperature, p.top_p, p.top_k, p.seed, p.stop, p.stop_token_ids) == (
        0.5, 0.9, 7, 3, ("x",), (9,),
    )  # fmt: skip
    assert p.max_tokens == 4 and p.ignore_eos


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"tools": TOOLS, "tool_choice": "required"}, "tool_choice"),
        (
            {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "x"}}},
            "tool_choice",
        ),
        ({"tool_choice": "auto"}, "tool_choice"),
        ({"tools": []}, "tools"),
        ({"tools": TOOLS, "parallel_tool_calls": False}, "parallel_tool_calls"),
    ],
)
def test_unsupported_tool_options_are_400(body, param):
    client, _, _ = _client()
    r = _post(client, **body)
    assert r.status_code == 400 and r.json()["error"]["param"] == param


def test_tools_are_400_when_the_server_has_no_tool_call_parser():
    client, _, _ = _client(parsers=OutputParsers())
    r = _post(client, tools=TOOLS)
    assert r.status_code == 400 and "--tool-call-parser" in r.json()["error"]["message"]
    assert _post(client, tools=TOOLS, tool_choice="none").status_code == 200


def test_a_failure_inside_the_stream_is_an_error_event_in_the_same_shape():
    client, scheduler, _ = _client(["a", "b", "c"], prompt="")
    scheduler.fail_after = 1
    events = _events_until_error(_post(client, stream=True))
    assert (
        events[-1]["error"]["code"] == 500 and events[-1]["error"]["type"] == "InternalServerError"
    )


def _events_until_error(response) -> list[dict]:
    lines = [ln for ln in response.text.splitlines() if ln.startswith("data: ")]
    return [json.loads(ln.removeprefix("data: ")) for ln in lines]


def test_an_internal_error_is_500_in_the_same_shape():
    client, scheduler, _ = _client(["a"], prompt="")
    scheduler.fail_after = 0
    r = _post(client)
    assert r.status_code == 500
    assert r.json() == {
        "error": {
            "message": "internal server error",
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def test_unknown_routes_are_404_in_the_same_shape():
    client, _, _ = _client()
    r = client.get("/v1/nothing")
    assert r.status_code == 404 and r.json()["error"]["code"] == 404

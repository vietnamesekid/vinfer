"""qwen3_coder (Qwen3.5 XML) and hermes (Qwen3 JSON) tool-call parsers, split anywhere."""

from __future__ import annotations

import json

import pytest

from tests.unit.server.splits import assert_same_for_every_split
from vinfer.server.parsers.base import merge_calls
from vinfer.server.parsers.hermes import HermesToolCallParser
from vinfer.server.parsers.qwen3_coder import Qwen3CoderToolCallParser
from vinfer.server.parsers.tool_text import coerce_value

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "scale": {"type": "number"},
                    "tags": {"type": "array"},
                    "opts": {"type": ["object", "null"]},
                },
            },
        },
    }
]


def _collect(deltas):
    content = "".join(d.content for d in deltas)
    pieces = [c for d in deltas for c in d.calls]
    calls = merge_calls(pieces)
    # The first piece of each call carries its id and name; later pieces only arguments.
    for i, call in enumerate(calls):
        mine = [c for c in pieces if c.index == i]
        assert (mine[0].id, mine[0].name) == (call.id, call.name)
        assert all(c.id is None and c.name is None for c in mine[1:])
    return content, [(c.name, json.loads(c.arguments)) for c in calls]


def _parse(cls, text, tools=TOOLS):
    streamed = assert_same_for_every_split(lambda: cls(tools), text, _collect)
    content, calls = cls(tools).parse(text)
    assert streamed == (content, [(c.name, json.loads(c.arguments)) for c in calls])
    assert len({c.id for c in calls}) == len(calls)
    assert all(c.id.startswith("chatcmpl-tool-") for c in calls)
    return streamed


def _xml(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No tools needed, 1 < 2.", ("No tools needed, 1 < 2.", [])),
        (
            "Let me check.\n\n" + _xml("get_weather", city="Hà Nội", days=3),
            ("Let me check.", [("get_weather", {"city": "Hà Nội", "days": 3})]),
        ),
        # Typed by the schema; a value that does not fit its type stays a string.
        (
            _xml("get_weather", metric="True", scale="1.5", tags='["a", 1]', opts="null"),
            ("", [("get_weather", {"metric": True, "scale": 1.5, "tags": ["a", 1], "opts": None})]),
        ),
        (_xml("get_weather", days="soon"), ("", [("get_weather", {"days": "soon"})])),
        # Multi-line values keep their inner newlines; only the markup newlines go.
        (
            _xml("get_weather", city="line 1\n\nline 2\n"),
            ("", [("get_weather", {"city": "line 1\n\nline 2\n"})]),
        ),
        # Unknown tool or parameter: values are strings.
        (_xml("other", n="5"), ("", [("other", {"n": "5"})])),
        # Two calls, newline between them is markup.
        (
            _xml("get_weather", days=1) + "\n" + _xml("get_weather", days=2),
            ("", [("get_weather", {"days": 1}), ("get_weather", {"days": 2})]),
        ),
        # No parameters.
        ("<tool_call>\n<function=ping>\n</function>\n</tool_call>", ("", [("ping", {})])),
    ],
)
def test_qwen3_coder(text, expected):
    assert _parse(Qwen3CoderToolCallParser, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A bare <function= without <tool_call>.
        (
            "<function=ping>\n<parameter=a>\nx\n</parameter>\n</function>",
            ("", [("ping", {"a": "x"})]),
        ),
        # A parameter left open is ended by the next one, or by </function>.
        (
            "<tool_call>\n<function=f>\n<parameter=a>\n1\n<parameter=b>\n2\n</function>\n</tool_call>",
            ("", [("f", {"a": "1", "b": "2"})]),
        ),
        # </function> missing: </tool_call> closes the call.
        (
            "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</tool_call>",
            ("", [("f", {"a": "1"})]),
        ),
        # Two functions in one <tool_call>.
        (
            "<tool_call>\n<function=f>\n</function>\n<function=g>\n</function>\n</tool_call>",
            ("", [("f", {}), ("g", {})]),
        ),
        # Cut off by max_tokens inside a value: what was written is the value.
        ("<tool_call>\n<function=f>\n<parameter=a>\npartial", ("", [("f", {"a": "partial"})])),
        # Cut off inside the name: no call.
        ("Hi <tool_call>\n<function=get_wea", ("Hi", [])),
        # Text after a call is content.
        (_xml("ping") + "\nDone.", ("Done.", [("ping", {})])),
        # A partial tag at the end of plain text is text.
        ("a <tool_ca", ("a <tool_ca", [])),
    ],
)
def test_qwen3_coder_malformed_and_cut_off_output(text, expected):
    assert _parse(Qwen3CoderToolCallParser, text) == expected


def test_qwen3_coder_streams_the_name_before_the_arguments():
    p = Qwen3CoderToolCallParser(TOOLS)
    first = p.feed("<tool_call>\n<function=get_weather>\n<parameter=days>\n")
    assert [(c.index, c.name, c.arguments) for c in first.calls] == [(0, "get_weather", "")]
    second = p.feed("2\n</parameter>\n")
    assert [c.arguments for c in second.calls] == ['{"days": 2']
    third = p.feed("</function>\n</tool_call>")
    assert [c.arguments for c in third.calls] == ["}"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", ("plain", [])),
        (
            'Sure.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Huế"}}'
            "\n</tool_call>",
            ("Sure.", [("get_weather", {"city": "Huế"})]),
        ),
        (
            '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "b", "arguments": "{\\"x\\": 1}"}\n</tool_call>',
            ("", [("a", {}), ("b", {"x": 1})]),
        ),
        # Not a valid call: kept as content, as written.
        ("<tool_call>\nnot json\n</tool_call>", ("<tool_call>\nnot json\n</tool_call>", [])),
        # Cut off: a complete JSON object still counts, anything else is content.
        ('<tool_call>\n{"name": "a", "arguments": {}}', ("", [("a", {})])),
        ('<tool_call>\n{"name": "a", "argu', ('<tool_call>\n{"name": "a", "argu', [])),
    ],
)
def test_hermes(text, expected):
    assert _parse(HermesToolCallParser, text) == expected


@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        ("5", None, "5"),
        ("5", {"type": "integer"}, 5),
        ("5", {"type": "number"}, 5),
        ("5.0", {"type": "number"}, 5.0),
        ("x", {"type": "integer"}, "x"),
        ("null", {"type": ["string", "null"]}, None),
        ("null", {"type": "string"}, "null"),
        ("FALSE", {"type": "boolean"}, False),
        ('{"a": 1}', {"anyOf": [{"type": "object"}, {"type": "string"}]}, {"a": 1}),
        ("[1", {"type": "array"}, "[1"),
        ("red", {"enum": ["red", "blue"]}, "red"),
        ("nan", {"type": "number"}, "nan"),
    ],
)
def test_values_are_typed_by_their_schema(value, schema, expected):
    got = coerce_value(value, schema)
    assert got == expected and type(got) is type(expected)

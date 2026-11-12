"""Parser registry, the model-type mapping, tag scanning, and the resolver chaining parsers."""

from __future__ import annotations

import pytest

from tests.unit.server.splits import assert_same_for_every_split
from vinfer.server.parsers import (
    MODEL_PARSERS,
    REASONING_PARSERS,
    TOOL_CALL_PARSERS,
    OutputParsers,
    resolve_parser_names,
)
from vinfer.server.parsers.base import ToolCallDelta, find_tag, merge_calls
from vinfer.server.parsers.resolver import OutputDelta


def test_find_tag_returns_the_earliest_tag_or_what_is_safe_to_release():
    assert find_tag("ab<x>cd<y>", ("<y>", "<x>")) == (2, "<x>")
    assert find_tag("abc<t", ("<tag>",)) == (3, None)  # "<t" may become "<tag>"
    assert find_tag("abc<q", ("<tag>",)) == (5, None)
    assert find_tag("", ("<tag>",)) == (0, None)
    assert find_tag("a</", ("</think>", "<tool_call>")) == (1, None)


def test_registries_know_every_parser_the_mapping_names():
    for reasoning, tool in MODEL_PARSERS.values():
        assert reasoning in REASONING_PARSERS.names() and tool in TOOL_CALL_PARSERS.names()
    assert "none" in REASONING_PARSERS.names() and "none" in TOOL_CALL_PARSERS.names()


def test_auto_follows_the_model_type_and_explicit_names_win():
    assert resolve_parser_names("auto", "auto", "qwen3_5") == ("qwen3", "qwen3_coder")
    assert resolve_parser_names("auto", "auto", "qwen3") == ("qwen3", "hermes")
    assert resolve_parser_names("auto", "auto", "unknown_family") == ("none", "none")
    assert resolve_parser_names("none", "hermes", "qwen3_5") == ("none", "hermes")


def test_unknown_names_fail_with_the_available_ones():
    with pytest.raises(ValueError, match="available"):
        resolve_parser_names("qwen9", "auto", "qwen3")
    with pytest.raises(ValueError, match="tool-call parser"):
        OutputParsers("none", "nope")
    with pytest.raises(ValueError, match="registered twice"):
        REASONING_PARSERS.register("qwen3")(object)


def test_none_parsers_pass_text_through_unchanged():
    text = "<think>a</think> <tool_call>\n<function=f>\n</function>\n</tool_call>"
    out = OutputParsers().new("<think>\n", tools=[]).resolve(text)
    assert (out.content, out.reasoning, out.tool_calls) == (text, None, [])


def test_tool_calls_are_parsed_only_when_the_request_asks():
    parsers = OutputParsers("qwen3", "qwen3_coder")
    text = "<tool_call>\n<function=f>\n</function>\n</tool_call>"
    assert parsers.new("", tools=None).resolve(text).content == text
    assert parsers.new("", tools=[]).resolve(text).tool_calls[0].name == "f"


def test_delta_sum_keeps_one_entry_per_call_in_order():
    d = OutputDelta()
    d += OutputDelta(tool_calls=[ToolCallDelta(0, "id0", "f"), ToolCallDelta(0, arguments="{")])
    d += OutputDelta(tool_calls=[ToolCallDelta(0, arguments="}"), ToolCallDelta(1, "id1", "g")])
    assert [(c.index, c.id, c.name, c.arguments) for c in d.tool_calls] == [
        (0, "id0", "f", "{}"),
        (1, "id1", "g", ""),
    ]
    assert [(c.id, c.arguments) for c in merge_calls(d.tool_calls)] == [("id0", "{}"), ("id1", "")]


def _collect(deltas):
    total = OutputDelta()
    for d in deltas:
        total += d
    calls = [(c.name, c.arguments) for c in merge_calls(total.tool_calls)]
    return total.content, total.reasoning, calls


@pytest.mark.parametrize(
    ("prompt_tail", "text"),
    [
        (
            "<|im_start|>assistant\n<think>\n",
            "Plan: call f.\n</think>\n\nCalling.\n\n<tool_call>\n<function=f>\n"
            "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
        ),
        # Qwen3.5 sometimes calls a tool straight from its reasoning.
        (
            "<|im_start|>assistant\n<think>\n",
            "Plan.\n<tool_call>\n<function=f>\n<parameter=x>\n2\n</parameter>\n</function>",
        ),
        ("<|im_start|>assistant\n<think>\n\n</think>\n\n", "Just text, a < b."),
    ],
)
def test_resolver_gives_the_same_message_however_the_text_is_split(prompt_tail, text):
    parsers = OutputParsers("qwen3", "qwen3_coder")
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    streamed = assert_same_for_every_split(lambda: parsers.new(prompt_tail, tools), text, _collect)
    whole = parsers.new(prompt_tail, tools).resolve(text)
    calls = [(c.name, c.arguments) for c in whole.tool_calls]
    assert streamed == (whole.content, whole.reasoning or "", calls)

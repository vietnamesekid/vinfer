"""Qwen3 / Qwen3.5 reasoning parser: start state from the prompt, tags split anywhere."""

from __future__ import annotations

import pytest

from tests.unit.server.splits import assert_same_for_every_split
from vinfer.server.parsers.base import ReasoningStart
from vinfer.server.parsers.think import Qwen3ReasoningParser

# Ends of prompts rendered by the real templates (generation prompt of the last turn).
QWEN35_THINKING = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
QWEN35_NOT_THINKING = (
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
QWEN3_DEFAULT = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
# An earlier assistant turn closed its reasoning; the new turn has no tag yet.
HISTORY = "<|im_start|>assistant\n<think>\nx\n</think>\n\nok<|im_end|>\n" + QWEN3_DEFAULT


@pytest.mark.parametrize(
    ("tail", "state"),
    [
        (QWEN35_THINKING, ReasoningStart.REASONING),
        (QWEN35_NOT_THINKING, ReasoningStart.CONTENT),
        (QWEN3_DEFAULT, ReasoningStart.UNDECIDED),
        (HISTORY, ReasoningStart.UNDECIDED),
        ("", ReasoningStart.UNDECIDED),
    ],
)
def test_start_state_comes_from_the_end_of_the_prompt(tail, state):
    assert Qwen3ReasoningParser(tail).state is state


def _collect(deltas):
    return "".join(d.reasoning for d in deltas), "".join(d.content for d in deltas)


def _parse(tail, text, implicit_ends=()):
    new = lambda: Qwen3ReasoningParser(tail, implicit_ends)  # noqa: E731
    streamed = assert_same_for_every_split(new, text, _collect)
    reasoning, content = Qwen3ReasoningParser(tail, implicit_ends).parse(text)
    assert streamed == (reasoning or "", content)
    return reasoning, content


@pytest.mark.parametrize(
    ("tail", "text", "expected"),
    [
        # Qwen3.5, thinking on: the output starts inside reasoning, without a tag.
        (QWEN35_THINKING, "Two plus two.\n</think>\n\nIt is 4.", ("Two plus two.", "It is 4.")),
        # Qwen3.5, thinking off: everything is content, even a stray end tag is dropped.
        (QWEN35_NOT_THINKING, "It is 4.", (None, "It is 4.")),
        (QWEN35_NOT_THINKING, "It is</think> 4.", (None, "It is 4.")),
        # Qwen3 thinks by default and writes the start tag itself.
        (QWEN3_DEFAULT, "<think>\nHmm.\n</think>\n\nYes.", ("Hmm.", "Yes.")),
        (QWEN3_DEFAULT, "\n<think>\nHmm.\n</think>\n\nYes.", ("Hmm.", "Yes.")),
        # A model that does not reason: plain content, leading whitespace kept.
        (QWEN3_DEFAULT, "  Hello <b>", (None, "  Hello <b>")),
        (HISTORY, "<thin", (None, "<thin")),
        # A repeated start tag inside reasoning is dropped.
        (QWEN35_THINKING, "a<think>b</think>c", ("ab", "c")),
        # Cut off by max_tokens inside reasoning: all of it is reasoning.
        (QWEN35_THINKING, "still thinking </thi", ("still thinking </thi", "")),
        # The end tag split right after the reasoning, then nothing.
        (QWEN35_THINKING, "r\n</think>\n\n", ("r", "")),
        (QWEN35_THINKING, "", ("", "")),
        # Non-ASCII text around the tags.
        (QWEN35_THINKING, "Nghĩ 🙂\n</think>\n\nTrả lời", ("Nghĩ 🙂", "Trả lời")),
    ],
)
def test_reasoning_and_content(tail, text, expected):
    assert _parse(tail, text) == expected


def test_a_tool_call_ends_reasoning_and_stays_in_the_content():
    text = "I should look it up.\n<tool_call>\n<function=f>"
    assert _parse(QWEN35_THINKING, text, ("<tool_call>",)) == (
        "I should look it up.",
        "<tool_call>\n<function=f>",
    )
    # Without the implicit end it is just reasoning text.
    assert _parse(QWEN35_THINKING, text)[1] == ""


def test_whitespace_inside_reasoning_is_kept_and_only_its_edges_trimmed():
    assert _parse(QWEN35_THINKING, "\n a \n\n b \n</think>x") == ("a \n\n b", "x")

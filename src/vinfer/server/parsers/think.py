"""Reasoning in a ``<think> ... </think>`` block, the format of Qwen3, Qwen3.5 and Kimi K2.

Where the output starts is decided from the end of the prompt, the way vLLM does it
(``vllm/parser/engine/parser_engine.py``, ``is_reasoning_end``): the last start tag, end tag
or turn boundary in the prompt wins. A template that switches thinking off closes the block
in the prompt (``<think>\\n\\n</think>``), so the output is content; one that switches it on
opens it (``<think>\\n``), so the output starts inside reasoning without a tag. With neither in
the last turn the model opens the block itself, or does not reason at all: the output is
reasoning only if it starts with the start tag.

Whitespace around the reasoning (and right after the end tag) is markup, not text: it is
trimmed, as the chat templates do when they render a past turn.
"""

from __future__ import annotations

from collections.abc import Sequence

from vinfer.server.parsers.base import ReasoningDelta, ReasoningParser, ReasoningStart, find_tag
from vinfer.server.parsers.registry import REASONING_PARSERS


class ThinkTagReasoningParser(ReasoningParser):
    start_tag = "<think>"
    end_tag = "</think>"
    turn_boundaries: tuple[str, ...] = ()  # special tokens that start or end a turn

    def __init__(self, prompt_tail: str = "", implicit_ends: Sequence[str] = ()) -> None:
        super().__init__(prompt_tail, implicit_ends)
        self.state = self.start_state(prompt_tail)
        self._saw_reasoning = self.state is ReasoningStart.REASONING
        self._buf = ""
        self._strip_leading = True  # drop whitespace before the next text of this state

    def start_state(self, prompt_tail: str) -> ReasoningStart:
        positions = {
            ReasoningStart.REASONING: prompt_tail.rfind(self.start_tag),
            ReasoningStart.CONTENT: prompt_tail.rfind(self.end_tag),
            ReasoningStart.UNDECIDED: max(
                (prompt_tail.rfind(b) for b in self.turn_boundaries), default=-1
            ),
        }
        state, pos = max(positions.items(), key=lambda kv: kv[1])
        return state if pos >= 0 else ReasoningStart.UNDECIDED

    @property
    def saw_reasoning(self) -> bool:
        return self._saw_reasoning

    def feed(self, text: str) -> ReasoningDelta:
        self._buf += text
        out = ReasoningDelta()
        while self._step(out):
            pass
        return out

    def finish(self) -> ReasoningDelta:
        out = ReasoningDelta()
        rest, self._buf = self._buf, ""
        if self.state is ReasoningStart.REASONING:
            out.reasoning = rest.strip() if self._strip_leading else rest.rstrip()
        elif self._strip_leading and self.state is ReasoningStart.CONTENT:
            out.content = rest.lstrip()
        else:
            out.content = rest
        return out

    def _step(self, out: ReasoningDelta) -> bool:
        """Consume what can be decided from the buffer; ``True`` if the state changed."""
        if self.state is ReasoningStart.UNDECIDED:
            head = self._buf.lstrip()
            if head.startswith(self.start_tag):
                self._buf = head[len(self.start_tag) :]
                self.state, self._saw_reasoning, self._strip_leading = (
                    ReasoningStart.REASONING,
                    True,
                    True,
                )
                return True
            if self.start_tag.startswith(head):  # empty, whitespace, or a partial start tag
                return False
            self.state, self._strip_leading = ReasoningStart.CONTENT, False
            return True

        if self.state is ReasoningStart.REASONING:
            # A repeated start tag inside reasoning is dropped (as vLLM does).
            tags = (self.end_tag, self.start_tag, *self.implicit_ends)
            i, tag = find_tag(self._buf, tags)
            text = self._buf[:i]
            if self._strip_leading:
                text = text.lstrip()
                self._strip_leading = not text
            if tag is None:
                # Trailing whitespace may be the markup before the end tag: hold it.
                keep = len(text) - len(text.rstrip())
                out.reasoning += text[: len(text) - keep]
                self._buf = text[len(text) - keep :] + self._buf[i:]
                return False
            if tag == self.start_tag:
                out.reasoning += text
                self._buf = self._buf[i + len(tag) :]
                return True
            out.reasoning += text.rstrip()
            if tag == self.end_tag:
                self._buf = self._buf[i + len(tag) :]
                self._strip_leading = True
            else:  # an implicit end: the tag is content (a tool call)
                self._buf = self._buf[i:]
                self._strip_leading = False
            self.state = ReasoningStart.CONTENT
            return True

        # CONTENT: a repeated end tag is dropped (as vLLM does).
        i, tag = find_tag(self._buf, (self.end_tag,))
        text = self._buf[:i]
        if self._strip_leading:
            text = text.lstrip()
            self._strip_leading = not text
        out.content += text
        if tag is None:
            self._buf = self._buf[i:]
            return False
        self._buf = self._buf[i + len(tag) :]
        return True


@REASONING_PARSERS.register("qwen3")
class Qwen3ReasoningParser(ThinkTagReasoningParser):
    """Qwen3 (thinking on by default, the model writes ``<think>``) and Qwen3.5 (off by
    default; ``enable_thinking`` puts ``<think>`` in the prompt). Both tags are ordinary added
    tokens (``special: false``), so they survive ``skip_special_tokens``."""

    turn_boundaries = ("<|im_start|>", "<|im_end|>")

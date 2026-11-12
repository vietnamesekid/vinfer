"""The two parser interfaces, their deltas, and the tag scanning they share.

A parser is created per request and fed the generated text in pieces of any size (one token's
text, or the whole output). It may hold text back while it could still be the start of a tag,
so a tag split across tokens (``"</thi"`` + ``"nk>"``) is never shown to the client; the held
text is released by a later ``feed`` or by ``finish``. Feeding the whole text at once and
calling ``finish`` is the non-streaming path (``parse``), so both paths give the same result.
"""

from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


def find_tag(text: str, tags: Sequence[str]) -> tuple[int, str | None]:
    """The earliest complete tag in ``text``: ``(index, tag)``. Without one, ``(safe, None)``
    where ``text[:safe]`` cannot be part of any tag (the rest may still grow into one)."""
    best, found = len(text), None
    for tag in tags:
        i = text.find(tag)
        if i != -1 and (i < best or (i == best and found is not None and len(tag) > len(found))):
            best, found = i, tag
    if found is not None:
        return best, found
    return len(text) - _partial_tag_len(text, tags), None


def _partial_tag_len(text: str, tags: Sequence[str]) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of one of ``tags``."""
    longest = 0
    for tag in tags:
        for n in range(min(len(tag) - 1, len(text)), longest, -1):
            if text.endswith(tag[:n]):
                longest = n
                break
    return longest


# --- Reasoning ---------------------------------------------------------------------------------


@dataclass
class ReasoningDelta:
    reasoning: str = ""
    content: str = ""


class ReasoningStart(enum.Enum):
    """Where the output starts, decided from the end of the prompt."""

    REASONING = "reasoning"  # the prompt opened a reasoning block
    CONTENT = "content"  # the prompt closed it (thinking switched off)
    UNDECIDED = "undecided"  # the model opens it itself, or never does


class ReasoningParser(ABC):
    """Splits generated text into ``reasoning`` and ``content``.

    ``prompt_tail`` is the end of the rendered prompt with special tokens kept, so a parser can
    tell whether the template already opened or closed the reasoning block. ``implicit_ends``
    are tags that end reasoning without a closing tag and belong to the content (the tool-call
    parser's start tags: a model may call a tool straight from its reasoning).
    """

    def __init__(self, prompt_tail: str = "", implicit_ends: Sequence[str] = ()) -> None:
        self.implicit_ends = tuple(implicit_ends)

    @abstractmethod
    def feed(self, text: str) -> ReasoningDelta: ...

    @abstractmethod
    def finish(self) -> ReasoningDelta:
        """Release everything held back; the parser is done."""

    def parse(self, text: str) -> tuple[str | None, str]:
        """Whole output -> ``(reasoning, content)``; reasoning is ``None`` if there was none."""
        a, b = self.feed(text), self.finish()
        reasoning = a.reasoning + b.reasoning
        return (reasoning if self.saw_reasoning else None), a.content + b.content

    @property
    def saw_reasoning(self) -> bool:
        return False


class IdentityReasoningParser(ReasoningParser):
    """Everything is content."""

    def feed(self, text: str) -> ReasoningDelta:
        return ReasoningDelta(content=text)

    def finish(self) -> ReasoningDelta:
        return ReasoningDelta()


# --- Tool calls --------------------------------------------------------------------------------


@dataclass
class ToolCallDelta:
    """A piece of tool call ``index``: the first piece carries ``id`` and ``name``; the
    ``arguments`` pieces concatenate to a JSON object."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON object


@dataclass
class ToolDelta:
    content: str = ""
    calls: list[ToolCallDelta] = field(default_factory=list)


def new_tool_call_id() -> str:
    return f"chatcmpl-tool-{uuid.uuid4().hex}"


def merge_calls(deltas: Sequence[ToolCallDelta]) -> list[ToolCall]:
    """Assemble streamed pieces into complete calls, in index order."""
    calls: dict[int, ToolCall] = {}
    for d in deltas:
        call = calls.setdefault(d.index, ToolCall(d.id or "", d.name or "", ""))
        call.id = call.id or d.id or ""
        call.name = call.name or d.name or ""
        call.arguments += d.arguments
    return [calls[i] for i in sorted(calls)]


class ToolCallParser(ABC):
    """Extracts tool calls from the content (what is left after reasoning).

    ``tools`` are the request's tool definitions (OpenAI format), used to give parameter values
    their schema type. ``start_tags`` open a tool call; a reasoning parser ends on them.
    """

    start_tags: tuple[str, ...] = ()

    def __init__(self, tools: Sequence[dict[str, Any]] = ()) -> None:
        self.tools = list(tools)
        self._count = 0

    def _new_call(self, name: str) -> ToolCallDelta:
        """The first piece of the next tool call."""
        self._count += 1
        return ToolCallDelta(self._count - 1, new_tool_call_id(), name, "")

    def parameter_schemas(self, name: str) -> dict[str, Any]:
        """``properties`` of tool ``name``'s JSON schema (empty if unknown)."""
        for tool in self.tools:
            fn = tool.get("function") or {}
            if fn.get("name") == name:
                return (fn.get("parameters") or {}).get("properties") or {}
        return {}

    @abstractmethod
    def feed(self, text: str) -> ToolDelta: ...

    @abstractmethod
    def finish(self) -> ToolDelta:
        """Release everything held back and close an unfinished call; the parser is done."""

    def parse(self, text: str) -> tuple[str, list[ToolCall]]:
        """Whole content -> ``(content without the calls, calls)``."""
        a, b = self.feed(text), self.finish()
        return a.content + b.content, merge_calls(a.calls + b.calls)


class IdentityToolCallParser(ToolCallParser):
    """No tool calls: the content passes through."""

    def feed(self, text: str) -> ToolDelta:
        return ToolDelta(content=text)

    def finish(self) -> ToolDelta:
        return ToolDelta()

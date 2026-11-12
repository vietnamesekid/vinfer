"""Turn one request's generated text into a standard OpenAI message, whatever the model.

The text goes through the reasoning parser first; what it leaves as content goes through the
tool-call parser. The server sees only ``content``, ``reasoning`` and ``tool_calls``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vinfer.server.parsers.base import (
    ReasoningParser,
    ToolCall,
    ToolCallDelta,
    ToolCallParser,
    merge_calls,
)
from vinfer.server.parsers.registry import NONE, REASONING_PARSERS, TOOL_CALL_PARSERS


@dataclass
class OutputDelta:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCallDelta] = field(default_factory=list)

    def __iadd__(self, other: OutputDelta) -> OutputDelta:
        self.content += other.content
        self.reasoning += other.reasoning
        for call in other.tool_calls:  # one entry per call index within a delta
            same = next((c for c in self.tool_calls if c.index == call.index), None)
            if same is None:
                self.tool_calls.append(ToolCallDelta(**vars(call)))
            else:
                same.id, same.name = same.id or call.id, same.name or call.name
                same.arguments += call.arguments
        return self


@dataclass
class OutputMessage:
    content: str
    reasoning: str | None  # None: the output had no reasoning block
    tool_calls: list[ToolCall]


class OutputResolver:
    """Parsers for one request. Feed it text deltas, then ``finish``; or ``resolve`` a whole
    output."""

    def __init__(self, reasoning: ReasoningParser, tools: ToolCallParser) -> None:
        self.reasoning = reasoning
        self.tools = tools
        self.saw_tool_call = False

    def feed(self, text: str) -> OutputDelta:
        r = self.reasoning.feed(text)
        t = self.tools.feed(r.content)
        self.saw_tool_call |= bool(t.calls)
        out = OutputDelta(t.content, r.reasoning)
        out += OutputDelta(tool_calls=t.calls)
        return out

    def finish(self) -> OutputDelta:
        r = self.reasoning.finish()
        out = OutputDelta()
        for t in (self.tools.feed(r.content), self.tools.finish()):
            out += OutputDelta(t.content, "", t.calls)
        out.reasoning = r.reasoning
        self.saw_tool_call |= bool(out.tool_calls)
        return out

    def resolve(self, text: str) -> OutputMessage:
        out = self.feed(text)
        out += self.finish()
        reasoning = out.reasoning if self.reasoning.saw_reasoning else None
        return OutputMessage(out.content, reasoning, merge_calls(out.tool_calls))


@dataclass(frozen=True)
class OutputParsers:
    """The parsers chosen for a server (``ServingConfig``), resolved to names."""

    reasoning: str = NONE
    tool_call: str = NONE

    def __post_init__(self) -> None:
        REASONING_PARSERS.get(self.reasoning)
        TOOL_CALL_PARSERS.get(self.tool_call)

    @property
    def supports_tools(self) -> bool:
        return self.tool_call != NONE

    def new(self, prompt_tail: str, tools: Sequence[dict[str, Any]] | None) -> OutputResolver:
        """A resolver for one request. ``tools``: the request's tool definitions, or ``None``
        when tool calls are not to be parsed (no tools, or ``tool_choice: "none"``)."""
        tool_parser = TOOL_CALL_PARSERS.get(self.tool_call if tools is not None else NONE)(
            tools or ()
        )
        reasoning = REASONING_PARSERS.get(self.reasoning)(prompt_tail, tool_parser.start_tags)
        return OutputResolver(reasoning, tool_parser)

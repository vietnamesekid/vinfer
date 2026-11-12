"""Named parsers, and which ones a model family uses.

Adding a model family is a parser class registered under a name (if its format is new) and one
entry in ``MODEL_PARSERS``; the server never looks at the model type itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from vinfer.server.parsers.base import (
    IdentityReasoningParser,
    IdentityToolCallParser,
    ReasoningParser,
    ToolCallParser,
)

T = TypeVar("T")

AUTO, NONE = "auto", "none"


class ParserRegistry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._classes: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def wrap(cls: type[T]) -> type[T]:
            if name in self._classes or name == AUTO:
                raise ValueError(f"{self.kind} parser {name!r} registered twice")
            self._classes[name] = cls
            return cls

        return wrap

    def names(self) -> list[str]:
        return sorted(self._classes)

    def get(self, name: str) -> type[T]:
        if name not in self._classes:
            raise ValueError(f"unknown {self.kind} parser {name!r}; available: {self.names()}")
        return self._classes[name]


REASONING_PARSERS: ParserRegistry[ReasoningParser] = ParserRegistry("reasoning")
TOOL_CALL_PARSERS: ParserRegistry[ToolCallParser] = ParserRegistry("tool-call")

REASONING_PARSERS.register(NONE)(IdentityReasoningParser)
TOOL_CALL_PARSERS.register(NONE)(IdentityToolCallParser)

# model_type (config.json) -> (reasoning parser, tool-call parser) for "auto".
MODEL_PARSERS: dict[str, tuple[str, str]] = {
    "qwen3": ("qwen3", "hermes"),
    "qwen3_5": ("qwen3", "qwen3_coder"),
    "qwen3_5_text": ("qwen3", "qwen3_coder"),
}


def resolve_parser_names(reasoning: str, tool_call: str, model_type: str | None) -> tuple[str, str]:
    """Replace ``"auto"`` by the model family's parser (``"none"`` for an unknown family) and
    check that the names exist."""
    auto = MODEL_PARSERS.get(model_type or "", (NONE, NONE))
    reasoning = auto[0] if reasoning == AUTO else reasoning
    tool_call = auto[1] if tool_call == AUTO else tool_call
    REASONING_PARSERS.get(reasoning)
    TOOL_CALL_PARSERS.get(tool_call)
    return reasoning, tool_call

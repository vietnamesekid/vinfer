"""Output resolvers: each model family's generated text -> OpenAI ``content``,
``reasoning_content`` and ``tool_calls``.

Two kinds of parser, each a registry of named classes (``registry.py``), chosen per server by
``ServingConfig`` (``"auto"`` = from the checkpoint's ``model_type`` via ``MODEL_PARSERS``):

- ``ReasoningParser`` splits reasoning from content; it starts in the state the prompt left
  (a template may open or close the reasoning block itself).
- ``ToolCallParser`` extracts tool calls from that content.

``OutputResolver`` chains them for one request. Both work on text deltas of any size and hold
back a possible partial tag, so streaming and non-streaming give the same message.
"""

from vinfer.server.parsers import hermes, qwen3_coder, think  # noqa: F401  (registers parsers)
from vinfer.server.parsers.base import (
    ReasoningParser,
    ToolCall,
    ToolCallDelta,
    ToolCallParser,
)
from vinfer.server.parsers.registry import (
    MODEL_PARSERS,
    REASONING_PARSERS,
    TOOL_CALL_PARSERS,
    resolve_parser_names,
)
from vinfer.server.parsers.resolver import OutputDelta, OutputMessage, OutputParsers, OutputResolver

__all__ = [
    "MODEL_PARSERS",
    "REASONING_PARSERS",
    "TOOL_CALL_PARSERS",
    "OutputDelta",
    "OutputMessage",
    "OutputParsers",
    "OutputResolver",
    "ReasoningParser",
    "ToolCall",
    "ToolCallDelta",
    "ToolCallParser",
    "resolve_parser_names",
]

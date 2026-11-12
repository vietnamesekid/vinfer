"""OpenAI chat-completions request and response schemas (pydantic).

Requests accept unknown fields, as OpenAI-compatible servers do (clients send vendor extras),
but a known field that vinfer does not implement is rejected when it asks for a non-default
behaviour (``unsupported_fields``), never silently ignored.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")


class FunctionDefinition(_Model):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class Tool(_Model):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedToolChoice(_Model):
    type: Literal["function"] = "function"
    function: dict[str, Any]


class FunctionCallIn(_Model):
    name: str
    arguments: str | dict[str, Any] | None = None


class ToolCallIn(_Model):
    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCallIn


def _text_of_parts(content: Any) -> Any:
    """Content as an array of parts -> its text parts joined by newlines (as vLLM does).
    Other parts (images, audio) are not supported."""
    if not isinstance(content, list):
        return content
    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text" and "text" in part:
            texts.append(part["text"])
        else:
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise ValueError(f"only text content parts are supported, got {kind!r}")
    return "\n".join(texts)


class ChatMessage(_Model):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCallIn] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _join_parts(cls, value: Any) -> Any:
        return _text_of_parts(value)

    def for_template(self) -> dict[str, Any]:
        """The message as chat templates expect it: tool-call ``arguments`` as a dict (the API
        sends a JSON string; malformed JSON becomes ``{}`` so the conversation can go on, as
        vLLM does), no unset fields."""
        msg = self.model_dump(exclude_none=True)
        for call in msg.get("tool_calls") or []:
            args = call["function"].get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {}
            call["function"]["arguments"] = args if isinstance(args, dict) else {}
        return msg


class StreamOptions(_Model):
    include_usage: bool = False
    continuous_usage_stats: bool = False  # vLLM: usage on every chunk (needs include_usage)


class ChatCompletionRequest(_Model):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None  # vLLM/SGLang extension
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None  # vLLM/SGLang extension
    ignore_eos: bool = False  # vLLM/SGLang extension
    n: int = 1
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None  # vLLM/SGLang extension
    user: str | None = None  # accepted, unused

    @field_validator("stop_token_ids", mode="before")
    @classmethod
    def _no_bools(cls, value: Any) -> Any:
        if isinstance(value, list) and any(isinstance(v, bool) for v in value):
            raise ValueError("stop_token_ids must be a list of integers")
        return value


# Known OpenAI / vLLM fields vinfer does not implement, with the values that mean "off".
UNSUPPORTED_DEFAULTS: dict[str, tuple[Any, ...]] = {
    "logprobs": (None, False),
    "top_logprobs": (None, 0),
    "presence_penalty": (None, 0, 0.0),
    "frequency_penalty": (None, 0, 0.0),
    "repetition_penalty": (None, 1, 1.0),
    "logit_bias": (None, {}),
    "response_format": (None, {"type": "text"}),
    "functions": (None, []),
    "function_call": (None, "none"),
    "reasoning_effort": (None,),
    "min_tokens": (None, 0),
    "min_p": (None, 0, 0.0),
    "best_of": (None, 1),
    "use_beam_search": (None, False),
    "echo": (None, False),
    "prompt_logprobs": (None,),
    "guided_json": (None,),
    "guided_regex": (None,),
    "guided_choice": (None,),
    "guided_grammar": (None,),
    "structured_outputs": (None,),
    "continue_final_message": (None, False),
    "add_generation_prompt": (None, True),
    "audio": (None,),
    "modalities": (None, ["text"]),
    "prediction": (None,),
    "web_search_options": (None,),
}


def unsupported_fields(request: ChatCompletionRequest) -> list[str]:
    """Fields of ``request`` asking for something vinfer does not implement."""
    extra = request.model_extra or {}
    bad = [k for k, ok in UNSUPPORTED_DEFAULTS.items() if k in extra and extra[k] not in ok]
    if request.n != 1:
        bad.append("n")
    return bad


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class FunctionCallOut(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallOut(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCallOut


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    # The same text under both names: DeepSeek-style clients read reasoning_content, current
    # vLLM clients read reasoning.
    reasoning_content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallOut] = Field(default_factory=list)


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = None


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class DeltaToolCall(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallOut


class DeltaMessage(BaseModel):
    """Only the fields that are set are sent."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None  # also sent as ``reasoning`` (see ResponseMessage)
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None

    @field_serializer("delta")
    def _set_fields_only(self, delta: DeltaMessage) -> dict[str, Any]:
        return delta.model_dump(exclude_none=True)


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vinfer"
    max_model_len: int | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(BaseModel):
    """OpenAI's error body, as vLLM sends it: ``{"error": {message, type, param, code}}``."""

    error: ErrorInfo

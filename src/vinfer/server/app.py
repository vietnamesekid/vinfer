"""OpenAI-compatible ``/v1/chat/completions`` (streaming and not), on top of a scheduler.

Streaming follows what vLLM and SGLang send, which the bench harness measures: an opening chunk
with only the ``role``, then **one chunk per generated token** (its delta may be empty, e.g.
while text is held back as a possible stop string or a partial tag), the last token chunk
carrying ``finish_reason``, then a usage chunk if ``stream_options.include_usage`` is set, then
``[DONE]``. With ``include_usage`` every other chunk has ``"usage": null``, or the usage so far
with ``continuous_usage_stats`` (a vLLM extension).

The generated text goes through the output resolver (``vinfer.server.parsers``), which splits
it into ``content``, ``reasoning_content`` and ``tool_calls`` for any model family; this module
never looks at the model type.

Extensions accepted in the body, as vLLM/SGLang do: ``top_k``, ``ignore_eos``,
``stop_token_ids``, ``chat_template_kwargs``. Errors are OpenAI's ``{"error": {...}}``
(``errors.py``); every request error is found before the response starts.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from vinfer.runtime import SamplingParams, Step, check_request
from vinfer.runtime.tokenizer import Tokenizer
from vinfer.server.errors import APIError, error_body, install_error_handlers
from vinfer.server.parsers import OutputDelta, OutputParsers
from vinfer.server.protocol import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    DeltaMessage,
    DeltaToolCall,
    FunctionCallOut,
    ModelCard,
    ModelList,
    ResponseMessage,
    ToolCallOut,
    Usage,
    unsupported_fields,
)

log = logging.getLogger("vinfer.server")

PROMPT_TAIL_TOKENS = 32  # enough for the last turn's header and a reasoning tag or two


class Scheduler(Protocol):
    @property
    def max_seq_len(self) -> int: ...

    def stream(self, prompt_ids: list[int], params: SamplingParams) -> Any: ...


def sampling_params(req: ChatCompletionRequest, default_max_tokens: int) -> SamplingParams:
    """``default_max_tokens`` applies when the body gives neither ``max_completion_tokens`` nor
    ``max_tokens`` (the rest of the context, as vLLM does). An explicit value is used as given,
    so 0 is rejected rather than replaced."""
    stop = req.stop or ()
    max_tokens = req.max_completion_tokens
    if max_tokens is None:
        max_tokens = req.max_tokens
    if max_tokens is None:
        max_tokens = default_max_tokens
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0 if req.temperature is None else req.temperature,
        top_k=req.top_k or 0,
        top_p=1.0 if req.top_p is None else req.top_p,
        seed=req.seed,
        stop=(stop,) if isinstance(stop, str) else tuple(stop),
        stop_token_ids=tuple(req.stop_token_ids or ()),
        ignore_eos=req.ignore_eos,
    )


@dataclass
class _Prepared:
    prompt: list[int]
    params: SamplingParams
    tools: list[dict[str, Any]] | None  # tool definitions to parse calls against, or None


def _tools_to_parse(req: ChatCompletionRequest, parsers: OutputParsers) -> bool:
    choice = req.tool_choice
    if req.tools is not None and not req.tools:
        raise APIError("tools must not be empty; omit the field instead", param="tools")
    if choice is not None and not isinstance(choice, str):
        raise APIError("a named tool_choice is not supported", param="tool_choice")
    if choice == "required":
        raise APIError('tool_choice "required" is not supported', param="tool_choice")
    if choice not in (None, "none") and not req.tools:
        raise APIError(f'tool_choice "{choice}" needs tools', param="tool_choice")
    if req.parallel_tool_calls is False:
        raise APIError("parallel_tool_calls=false is not supported", param="parallel_tool_calls")
    parse = bool(req.tools) and choice != "none"
    if parse and not parsers.supports_tools:
        raise APIError(
            "tool calling is not enabled for this model (see --tool-call-parser)", param="tools"
        )
    return parse


def _prepare(
    req: ChatCompletionRequest, tokenizer: Tokenizer, parsers: OutputParsers, max_seq_len: int
) -> _Prepared:
    """Everything that can make the request fail, checked before any response starts: errors
    are a 400, never a 500 or an error event inside a 200 stream."""
    if bad := unsupported_fields(req):
        raise APIError(f"unsupported parameter(s): {', '.join(bad)}", param=bad[0])
    parse_tools = _tools_to_parse(req, parsers)
    tools = [t.model_dump(exclude_none=True) for t in req.tools] if req.tools else None
    messages = [m.for_template() for m in req.messages]
    try:
        prompt = tokenizer.chat_ids(messages, tools, **(req.chat_template_kwargs or {}))
    except (TypeError, ValueError) as exc:
        raise APIError(str(exc), param="messages") from exc
    try:
        params = sampling_params(req, max(1, max_seq_len - len(prompt) + 1))
        check_request(len(prompt), params, max_seq_len)
    except ValueError as exc:
        raise APIError(str(exc)) from exc
    return _Prepared(prompt, params, tools if parse_tools else None)


def _delta_message(d: OutputDelta) -> DeltaMessage:
    """Always some output field, so every token's chunk counts as output (``content: ""``)."""
    calls = [
        DeltaToolCall(
            index=c.index,
            id=c.id,
            type="function" if c.id else None,
            function=FunctionCallOut(name=c.name, arguments=c.arguments),
        )
        for c in d.tool_calls
    ]
    empty = not (d.reasoning or calls)
    return DeltaMessage(
        content=d.content if d.content or empty else None,
        reasoning_content=d.reasoning or None,
        reasoning=d.reasoning or None,
        tool_calls=calls or None,
    )


def create_app(
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    model_name: str,
    parsers: OutputParsers | None = None,
) -> FastAPI:
    """``parsers``: how generated text becomes a message (default: raw text as content)."""
    parsers = parsers or OutputParsers()
    started = int(time.time())
    app = FastAPI(title="vinfer")
    install_error_handlers(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        card = ModelCard(id=model_name, created=started, max_model_len=scheduler.max_seq_len)
        return ModelList(data=[card]).model_dump()

    @app.post("/v1/chat/completions")
    async def chat(req: ChatCompletionRequest):
        p = _prepare(req, tokenizer, parsers, scheduler.max_seq_len)
        tail = tokenizer.decode(p.prompt[-PROMPT_TAIL_TOKENS:], skip_special_tokens=False)
        resolver = parsers.new(tail, p.tools)
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        steps = scheduler.stream(p.prompt, p.params)

        def usage(n: int) -> Usage:
            return Usage(
                prompt_tokens=len(p.prompt), completion_tokens=n, total_tokens=len(p.prompt) + n
            )

        def finish_reason(step: Step) -> str | None:
            if step.finish_reason == "stop" and resolver.saw_tool_call:
                return "tool_calls"
            return step.finish_reason

        if req.stream:
            opts = req.stream_options
            include_usage = bool(opts and opts.include_usage)
            continuous = include_usage and bool(opts and opts.continuous_usage_stats)

            def chunk(delta: DeltaMessage, n: int, finish: str | None = None) -> str:
                event = ChatCompletionChunk(
                    id=rid,
                    created=created,
                    model=model_name,
                    choices=[ChunkChoice(delta=delta, finish_reason=finish)],
                    usage=usage(n) if continuous else None,
                ).model_dump()
                if not include_usage:
                    del event["usage"]
                return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            async def events():
                yield chunk(DeltaMessage(role="assistant", content=""), 0)
                n = 0
                try:
                    async for step in steps:
                        n += 1
                        delta = resolver.feed(step.text)
                        if step.finish_reason is not None:
                            delta += resolver.finish()
                        yield chunk(_delta_message(delta), n, finish_reason(step))
                except Exception as exc:  # mid-stream failure: report it as vLLM/SGLang do
                    log.exception("generation failed")
                    yield f"data: {json.dumps(error_body(str(exc), 500))}\n\n"
                    return
                finally:
                    await steps.aclose()  # client gone: cancel generation at the next token
                if include_usage:
                    end = ChatCompletionChunk(
                        id=rid, created=created, model=model_name, choices=[], usage=usage(n)
                    )
                    yield f"data: {end.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        text, n, last = [], 0, None
        try:
            async for step in steps:
                text.append(step.text)
                n += 1
                last = step
        finally:
            await steps.aclose()
        out = resolver.resolve("".join(text))
        calls = [
            ToolCallOut(id=c.id, function=FunctionCallOut(name=c.name, arguments=c.arguments))
            for c in out.tool_calls
        ]
        message = ResponseMessage(
            content=out.content if out.content or not calls else None,
            reasoning_content=out.reasoning,
            reasoning=out.reasoning,
            tool_calls=calls,
        )
        response = ChatCompletion(
            id=rid,
            created=created,
            model=model_name,
            choices=[Choice(message=message, finish_reason=finish_reason(last) if last else None)],
            usage=usage(n),
        )
        return JSONResponse(response.model_dump())

    return app

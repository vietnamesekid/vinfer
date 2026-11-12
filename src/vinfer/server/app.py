"""OpenAI-compatible ``/v1/chat/completions`` (streaming and not), on top of a scheduler.

Streaming follows what vLLM and SGLang send, which the bench harness measures: an opening chunk
with only the ``role``, then **one chunk per generated token** (its text may be empty, e.g. while
held back as a possible stop string), the last token chunk carrying ``finish_reason``, then a
usage chunk if ``stream_options.include_usage`` is set, then ``[DONE]``.

Extensions accepted in the body, as vLLM/SGLang do: ``top_k``, ``ignore_eos``,
``stop_token_ids``, ``chat_template_kwargs``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from vinfer.runtime import SamplingParams, check_request
from vinfer.runtime.tokenizer import Tokenizer
from vinfer.scheduler.serial import SerialScheduler


def sampling_params(body: dict[str, Any], default_max_tokens: int) -> SamplingParams:
    """``default_max_tokens`` applies when the body gives neither ``max_completion_tokens`` nor
    ``max_tokens`` (the rest of the context, as vLLM does). An explicit value is used as given,
    so 0 is rejected rather than replaced."""
    if body.get("n", 1) != 1:
        raise ValueError("only n = 1 is supported")
    stop = body.get("stop") or ()
    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = default_max_tokens
    return SamplingParams(
        max_tokens=int(max_tokens),
        temperature=float(body.get("temperature", 1.0)),
        top_k=int(body.get("top_k") or 0),
        top_p=float(body.get("top_p", 1.0)),
        seed=body.get("seed"),
        stop=(stop,) if isinstance(stop, str) else tuple(stop),
        stop_token_ids=tuple(body.get("stop_token_ids") or ()),
        ignore_eos=bool(body.get("ignore_eos", False)),
    )


def create_app(scheduler: SerialScheduler, tokenizer: Tokenizer, model_name: str) -> FastAPI:
    app = FastAPI(title="vinfer")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict[str, Any]):
        try:
            prompt = tokenizer.chat_ids(
                body["messages"], **(body.get("chat_template_kwargs") or {})
            )
            # Everything that can make the request fail is checked here, before any response
            # starts: errors are a 400, never a 500 or an error event inside a 200 stream.
            params = sampling_params(body, max(1, scheduler.max_seq_len - len(prompt) + 1))
            check_request(len(prompt), params, scheduler.max_seq_len)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        steps = scheduler.stream(prompt, params)

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            event = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(event)}\n\n"

        def usage(n: int) -> dict[str, int]:
            total = len(prompt) + n
            return {"prompt_tokens": len(prompt), "completion_tokens": n, "total_tokens": total}

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            async def events():
                yield chunk({"role": "assistant", "content": ""})
                n = 0
                try:
                    async for step in steps:
                        n += 1
                        yield chunk({"content": step.text}, step.finish_reason)
                except Exception as exc:  # mid-stream failure: report it as vLLM/SGLang do
                    yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
                    return
                finally:
                    await steps.aclose()  # client gone: cancel generation at the next token
                if include_usage:
                    end = {"id": rid, "object": "chat.completion.chunk", "created": created,
                           "model": model_name, "choices": [], "usage": usage(n)}  # fmt: skip
                    yield f"data: {json.dumps(end)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        text, n, finish = [], 0, None
        try:
            async for step in steps:
                text.append(step.text)
                n += 1
                finish = step.finish_reason
        finally:
            await steps.aclose()
        return JSONResponse(
            {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(text)},
                        "finish_reason": finish,
                    }
                ],
                "usage": usage(n),
            }
        )

    return app

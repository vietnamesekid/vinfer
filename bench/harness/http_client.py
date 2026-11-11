"""Send one streaming chat-completions request and time every output chunk.

Timing follows ``async_request_openai_chat_completions`` in
``vllm/benchmarks/lib/endpoint_request_func.py`` (vLLM v0.33.0), so our numbers can be compared
with ``vllm bench serve`` (see ``validation.py``):

- ``start_time``: just before the HTTP request is sent, i.e. after any client-side queueing.
- ``ttft``: first chunk carrying output - ``start_time``.
- ``itl``: gaps between consecutive output chunks.
- ``latency``: last output chunk - ``start_time`` (the trailing usage chunk does not count).
- ``output_tokens``: from the server's ``usage`` (``stream_options.include_usage``).
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from bench.harness.workload import Request


@dataclass
class RequestTiming:
    """What one request measured. Times are in seconds."""

    request_id: str
    success: bool = False
    error: str = ""
    arrival_time: float = 0.0  # when the load generator released it, from the run start
    start_time: float = 0.0  # when it was sent, from the run start
    queue_time: float = 0.0  # start_time - arrival_time: waiting in the client
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    latency: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int | None = None  # if the server reports prefix-cache hits

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_payload(
    model: str, request: Request, extra_body: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The request body: exactly ``max_tokens`` greedy output tokens (``ignore_eos``), with
    usage reported at the end of the stream."""
    body = {
        "model": model,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "max_completion_tokens": request.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return {**body, **(extra_body or {})}


def _is_output(choice: dict[str, Any]) -> bool:
    """True if a streamed choice carries output (text, reasoning or a tool call).

    The opening chunk that only announces the ``role`` is not output.
    """
    delta = choice.get("delta") or {}
    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
        return True
    if delta.get("tool_calls") or delta.get("function_call"):
        return True
    # A token can decode to "" (e.g. a skipped special token); it still counts.
    return delta.get("content") == "" and "role" not in delta


def _parse_event(line: str) -> dict[str, Any] | None:
    """The JSON of a server-sent-events ``data:`` line; None for other lines and ``[DONE]``."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    return None if data == "[DONE]" else json.loads(data)


async def stream_chat_completion(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    request: Request,
    extra_body: dict[str, Any] | None = None,
) -> RequestTiming:
    """Send ``request`` to ``url`` and time it. Never raises: failures are recorded.

    A request succeeds only if the stream carried output, reported its usage, had no error
    event, and produced exactly ``request.max_tokens`` tokens. Anything else (a truncated
    output, an in-stream error) would distort the metrics, so it counts as failed.
    """
    timing = RequestTiming(request_id=request.id, prompt_tokens=request.prompt_tokens or 0)
    start = last_output = time.perf_counter()
    timing.start_time = start
    got_output = False
    try:
        payload = build_payload(model, request, extra_body)
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                timing.error = f"HTTP {response.status_code}: {body[:500]!r}"
                return timing
            async for line in response.aiter_lines():
                event = _parse_event(line)
                if event is None:
                    continue
                now = time.perf_counter()
                if "error" in event:  # vLLM / SGLang report mid-stream failures this way
                    timing.error = f"error event: {json.dumps(event['error'])[:500]}"
                    return timing
                choices = event.get("choices") or []
                if choices and _is_output(choices[0]):
                    if got_output:
                        timing.itl.append(now - last_output)
                    else:
                        got_output = True
                        timing.ttft = now - start
                    last_output = now
                if usage := event.get("usage"):
                    _read_usage(timing, usage)
    except Exception:
        timing.error = traceback.format_exc(limit=3)
        return timing

    timing.latency = last_output - start
    if not got_output:
        timing.error = "no output chunk received"
    elif timing.output_tokens == 0:
        timing.error = "no usage chunk: output length cannot be checked"
    elif timing.output_tokens != request.max_tokens:
        timing.error = f"output has {timing.output_tokens} tokens, expected {request.max_tokens}"
    else:
        timing.success = True
    return timing


def _read_usage(timing: RequestTiming, usage: dict[str, Any]) -> None:
    timing.output_tokens = int(usage.get("completion_tokens") or 0)
    if (prompt := usage.get("prompt_tokens")) is not None:
        timing.prompt_tokens = int(prompt)
    details = usage.get("prompt_tokens_details") or {}
    if (cached := details.get("cached_tokens")) is not None:
        timing.cached_prompt_tokens = int(cached)

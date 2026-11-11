"""Send many requests to a server: Poisson arrivals, a concurrency cap, in-order sessions.

The arrival process mirrors ``get_request`` in ``vllm/benchmarks/serve.py`` (vLLM v0.33.0) with
burstiness 1: exponential gaps at ``request_rate``, rescaled so the last arrival lands exactly at
``n / request_rate`` (vLLM does this to keep throughput stable across seeds). With
``request_rate = inf`` every request is released at t=0 and only the concurrency cap shapes the
load. The cap is a semaphore around the HTTP call, as in vLLM.

Sessions (W4): a turn is sent only after the previous turn of its session finished, so the
server has seen that prefix. Waiting for the previous turn counts as queue time, not TTFT.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from bench.harness.http_client import RequestTiming, stream_chat_completion
from bench.harness.workload import Request


def arrival_times(n: int, request_rate: float, seed: int) -> list[float]:
    """Seconds after the start at which each of ``n`` requests is released."""
    if n == 0:
        return []
    if math.isinf(request_rate):
        return [0.0] * n
    if request_rate <= 0:
        raise ValueError(f"request_rate must be > 0, got {request_rate}")
    gaps = np.random.default_rng(seed).exponential(scale=1.0 / request_rate, size=n)
    times = np.cumsum(gaps)
    times *= (n / request_rate) / times[-1]
    return times.tolist()


async def send_requests(
    base_url: str,
    model: str,
    requests: Sequence[Request],
    *,
    max_concurrency: int,
    request_rate: float = math.inf,
    seed: int = 0,
    extra_body: dict[str, Any] | None = None,
    timeout_s: float = 3600.0,
) -> tuple[list[RequestTiming], float]:
    """Send ``requests``; return their timings (in request order) and the run's wall time."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    arrivals = arrival_times(len(requests), request_rate, seed)
    slots = asyncio.Semaphore(max_concurrency)
    # One event per session turn, set when that turn finishes.
    turn_done = {(r.session_id, r.turn): asyncio.Event() for r in requests if r.session_id}

    limits = httpx.Limits(max_connections=max_concurrency + 8, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), limits=limits) as client:
        run_start = time.perf_counter()

        async def send(request: Request, arrival: float) -> RequestTiming:
            await asyncio.sleep(max(0.0, run_start + arrival - time.perf_counter()))
            previous_turn = turn_done.get((request.session_id, request.turn - 1))
            if previous_turn is not None:
                await previous_turn.wait()
            try:
                async with slots:
                    timing = await stream_chat_completion(client, url, model, request, extra_body)
            finally:
                if request.session_id:
                    turn_done[(request.session_id, request.turn)].set()
            timing.arrival_time = arrival
            timing.start_time -= run_start
            timing.queue_time = timing.start_time - arrival
            return timing

        timings = await asyncio.gather(*map(send, requests, arrivals))
        wall_time = time.perf_counter() - run_start
    return list(timings), wall_time

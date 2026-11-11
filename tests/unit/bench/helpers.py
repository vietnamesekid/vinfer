"""Helpers shared by the harness tests: requests, and a fake server with known delays."""

from __future__ import annotations

import asyncio

from bench.harness.fake_server import FakeServer
from bench.harness.load_generator import send_requests
from bench.harness.workload import Request

PREFILL_S, DECODE_S = 0.03, 0.004


def make_requests(n: int, max_tokens: int = 10) -> list[Request]:
    return [Request(str(i), [{"role": "user", "content": "a b c"}], max_tokens) for i in range(n)]


def send_to_fake_server(requests, **options):
    """``(server, timings, wall_time)`` after sending ``requests`` to a fresh fake server."""

    async def go():
        server = FakeServer(PREFILL_S, DECODE_S)
        port = await server.start()
        try:
            timings, wall_time = await send_requests(
                f"http://127.0.0.1:{port}", "m", requests, **options
            )
        finally:
            await server.stop()
        return server, timings, wall_time

    return asyncio.run(go())

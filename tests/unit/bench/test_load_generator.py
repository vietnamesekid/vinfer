"""Arrivals, concurrency cap, session order, and timings against the fake server."""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from bench.harness.fake_server import FakeServer
from bench.harness.http_client import build_payload
from bench.harness.load_generator import arrival_times, send_requests
from bench.harness.metrics import run_metrics
from bench.harness.workload import Request
from tests.unit.bench.helpers import DECODE_S, PREFILL_S, make_requests, send_to_fake_server


def test_arrivals_are_poisson_rescaled_and_seeded():
    times = arrival_times(1000, request_rate=10.0, seed=0)
    assert times == arrival_times(1000, request_rate=10.0, seed=0)
    assert times != arrival_times(1000, request_rate=10.0, seed=1)
    assert times[-1] == pytest.approx(100.0)  # n / rate, like vLLM
    gaps = np.diff(times)
    assert gaps.min() >= 0
    assert gaps.std() / gaps.mean() == pytest.approx(1.0, abs=0.15)  # exponential: CV ~ 1
    assert arrival_times(5, math.inf, 0) == [0.0] * 5


def test_timings_match_the_fake_server_delays():
    server, timings, wall_time = send_to_fake_server(make_requests(4), max_concurrency=1)
    assert all(t.success for t in timings), [t.error for t in timings]
    assert server.max_inflight == 1
    assert all(t.output_tokens == 10 and len(t.itl) == 9 for t in timings)
    m = run_metrics(timings, wall_time)
    # asyncio sleeps overshoot, never undershoot: generous upper bounds.
    assert PREFILL_S * 1000 <= m["ttft_ms"]["p50"] < PREFILL_S * 1000 + 25
    assert DECODE_S * 1000 <= m["tpot_ms"]["mean"] < DECODE_S * 1000 + 6
    assert m["output_tokens_per_s"] == pytest.approx(40 / wall_time)


def test_concurrency_cap_and_queue_time():
    server, timings, _ = send_to_fake_server(make_requests(12, max_tokens=5), max_concurrency=3)
    assert server.max_inflight == 3
    # All released at t=0 but capped: later requests wait in the client, outside TTFT.
    assert max(t.queue_time for t in timings) > PREFILL_S
    assert max(t.ttft for t in timings) < PREFILL_S + 0.05


def test_session_turns_are_sent_in_order():
    requests = [
        Request(f"{s}-{t}", [{"role": "user", "content": "x"}], 3, session_id=str(s), turn=t)
        for s in range(3)
        for t in range(3)
    ]
    _, timings, _ = send_to_fake_server(requests, max_concurrency=9)
    by_id = {t.request_id: t for t in timings}
    for s in range(3):
        for t in range(1, 3):
            previous, current = by_id[f"{s}-{t - 1}"], by_id[f"{s}-{t}"]
            assert current.start_time >= previous.start_time + previous.latency


@pytest.mark.parametrize("server_options", [{"short_by": 2}, {"error_after": 3}])
def test_truncated_or_errored_outputs_count_as_failed(server_options):
    """Only complete outputs are measured: a short output or an error event is a failure."""

    async def go():
        server = FakeServer(prefill_s=0.001, decode_s=0.001, **server_options)
        port = await server.start()
        try:
            request = Request("0", [{"role": "user", "content": "x"}], max_tokens=8)
            timings, _ = await send_requests(
                f"http://127.0.0.1:{port}", "m", [request], max_concurrency=1
            )
        finally:
            await server.stop()
        return timings[0]

    timing = asyncio.run(go())
    assert not timing.success and timing.error


def test_payload_always_forces_the_output_length():
    request = Request("0", [{"role": "user", "content": "x"}], 5)
    assert build_payload("m", request)["ignore_eos"] is True

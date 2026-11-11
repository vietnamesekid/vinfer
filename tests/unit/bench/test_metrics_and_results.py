"""Metric definitions, run aggregation, result files."""

from __future__ import annotations

import math

import pytest

from bench.harness import result_file
from bench.harness.http_client import RequestTiming, build_payload
from bench.harness.metrics import run_metrics
from tests.unit.bench.helpers import make_requests


def test_goodput_counts_requests_within_every_slo():
    timings = [
        RequestTiming("a", success=True, ttft=0.1, latency=1.1, output_tokens=11),  # TPOT 100ms
        RequestTiming("b", success=True, ttft=0.5, latency=0.6, output_tokens=11),  # TTFT 500ms
        RequestTiming("c", success=True, ttft=0.1, latency=0.1, output_tokens=1),  # TPOT 0
        RequestTiming("d", success=False),
    ]
    m = run_metrics(timings, wall_time=2.0, slo_ttft_ms=200, slo_tpot_ms=50)
    assert m["goodput_req_per_s"] == pytest.approx(0.5)  # only "c"
    assert m["failed"] == 1
    assert run_metrics(timings, 2.0)["goodput_req_per_s"] is None


def test_aggregate_runs_mean_and_sample_stddev():
    runs = [{"x": 1.0, "y": {"p50": 2.0}, "z": None}, {"x": 3.0, "y": {"p50": 4.0}, "z": None}]
    mean, std = result_file.aggregate_runs(runs)
    assert mean == {"x": 2.0, "y": {"p50": 3.0}, "z": None}
    assert std["x"] == pytest.approx(math.sqrt(2.0))


def test_payload_forces_greedy_exact_length_and_merges_extra_body():
    request = make_requests(1, max_tokens=7)[0]
    body = build_payload("m", request, {"seed": 1})
    assert body["max_tokens"] == body["max_completion_tokens"] == 7
    assert body["ignore_eos"] is True and body["temperature"] == 0.0
    assert body["stream_options"] == {"include_usage": True} and body["seed"] == 1


def test_hit_rate_counts_requests_without_cached_field_as_zero():
    def timing(cached):
        return RequestTiming("x", success=True, ttft=0.1, latency=0.2, output_tokens=2,
                             prompt_tokens=100, cached_prompt_tokens=cached)  # fmt: skip

    assert run_metrics([timing(None), timing(50)], 1.0)["prefix_cache_hit_rate"] == 0.25
    assert run_metrics([timing(None), timing(None)], 1.0)["prefix_cache_hit_rate"] is None

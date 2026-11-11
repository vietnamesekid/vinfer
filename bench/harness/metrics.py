"""Metrics of one run, defined like ``calculate_metrics`` in
``vllm/benchmarks/serve.py`` (vLLM v0.33.0):

- TPOT of a request = (latency - TTFT) / (output tokens - 1), for requests with > 1 output
  token. For goodput, a request with <= 1 output token has TPOT 0.
- Throughput = output tokens of successful requests / wall time of the run.
- Goodput = successful requests meeting every given SLO / wall time; None without an SLO.
- Prefix cache hit rate = cached prompt tokens / prompt tokens. None if no request reported
  cached tokens (the server does not report them); a request without the field counts as 0.

The result is a plain dict because it goes straight into the result JSON
(``bench/results/SCHEMA.md``). Times are in milliseconds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bench.harness.http_client import RequestTiming

MS = 1000.0
NAN = float("nan")


def _stats_ms(seconds: Sequence[float], *stats: str) -> dict[str, float]:
    """``{"mean": ..., "p50": ...}`` in milliseconds; NaN when there are no samples."""
    if not seconds:
        return dict.fromkeys(stats, NAN)
    arr = np.asarray(seconds) * MS
    return {s: float(arr.mean() if s == "mean" else np.percentile(arr, int(s[1:]))) for s in stats}


def _tpot(t: RequestTiming) -> float:
    return (t.latency - t.ttft) / (t.output_tokens - 1) if t.output_tokens > 1 else 0.0


def _goodput(
    ok: list[RequestTiming], wall_time: float, ttft_ms: float | None, tpot_ms: float | None
) -> float | None:
    if ttft_ms is None and tpot_ms is None:
        return None
    good = [
        t
        for t in ok
        if (ttft_ms is None or t.ttft * MS <= ttft_ms)
        and (tpot_ms is None or _tpot(t) * MS <= tpot_ms)
    ]
    return len(good) / wall_time


def run_metrics(
    timings: Sequence[RequestTiming],
    wall_time: float,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
) -> dict[str, Any]:
    ok = [t for t in timings if t.success]
    ttfts = [t.ttft for t in ok]
    tpots = [_tpot(t) for t in ok if t.output_tokens > 1]
    itls = [gap for t in ok for gap in t.itl]
    latencies = [t.latency for t in ok]
    queue = [t.queue_time for t in ok]
    prompt_tokens = sum(t.prompt_tokens for t in ok)

    hit_rate = None
    if prompt_tokens and any(t.cached_prompt_tokens is not None for t in ok):
        hit_rate = sum(t.cached_prompt_tokens or 0 for t in ok) / prompt_tokens

    return {
        "ttft_ms": _stats_ms(ttfts, "p50", "p95", "p99"),
        "tpot_ms": _stats_ms(tpots, "mean", "p50", "p99"),
        "itl_ms": _stats_ms(itls, "mean", "p50", "p99"),
        "e2el_ms": _stats_ms(latencies, "p50", "p99"),
        "queue_ms": _stats_ms(queue, "p50", "p99"),
        "output_tokens_per_s": sum(t.output_tokens for t in ok) / wall_time,
        "input_tokens_per_s": prompt_tokens / wall_time,
        "request_per_s": len(ok) / wall_time,
        "goodput_req_per_s": _goodput(ok, wall_time, slo_ttft_ms, slo_tpot_ms),
        "peak_vram_gb": None,  # filled in by the caller from gpu_memory
        "prefix_cache_hit_rate": hit_rate,
        "completed": len(ok),
        "failed": len(timings) - len(ok),
        "duration_s": wall_time,
    }

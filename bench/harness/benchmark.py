"""The ``run`` command: measure one server on one or more workloads over a concurrency sweep.

The server is started once for all workloads (model loading and graph capture are paid once);
the prefix cache is reset before every measured run, so workloads do not see each other.

For each concurrency level: warm up at that concurrency (see ``warmup_set``), then for each
run reset the prefix cache and send the workload. Each level becomes one result
file (``result_file.py``); per-request timings go to ``<out_dir>/raw/<run id>/``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

from bench.harness import environment, result_file
from bench.harness.config import REPO, BenchmarkConfig, HarnessError
from bench.harness.engines import RunningServer, reset_prefix_cache, start_server
from bench.harness.gpu_memory import PeakGpuMemory
from bench.harness.load_generator import send_requests
from bench.harness.metrics import run_metrics
from bench.harness.workload import (
    Request,
    load_workload,
    requests_per_run,
    select_requests,
    warmup_requests,
)


async def run_benchmark(cfg: BenchmarkConfig) -> list[Path]:
    """Measure every concurrency level of ``cfg``; return the result files written."""
    git = environment.git_state()
    if git.dirty and not cfg.allow_dirty:
        raise HarnessError("git tree is dirty: commit first, or pass --allow-dirty (unofficial)")
    # Load every workload before starting the server: a missing file fails in seconds.
    if cfg.workload_file is not None:
        workloads = [load_workload(path=cfg.workload_file)]
    else:
        workloads = [load_workload(name) for name in cfg.workloads]
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H%M%S.%f")[:-3]
    run_dir = cfg.out_dir / "raw" / f"{stamp}_{cfg.server.engine}-{cfg.server.config}"
    run_dir.mkdir(parents=True, exist_ok=True)

    written = []
    async with start_server(cfg.server, run_dir / "server.log") as server:
        for workload, all_requests in workloads:
            raw_dir = run_dir / workload
            raw_dir.mkdir(exist_ok=True)
            for concurrency in cfg.concurrency:
                requests = select_requests(
                    all_requests, requests_per_run(cfg.num_requests, concurrency)
                )
                per_run = await measure_level(cfg, server, requests, concurrency, raw_dir)
                result = build_result(cfg, server, workload, requests, concurrency, per_run, git)
                result["raw_dir"] = os.path.relpath(raw_dir, REPO)
                path = result_file.write(result, cfg.out_dir, stamp)
                problems = result["problems"]
                note = f" (unofficial: {'; '.join(problems)})" if problems else ""
                print(f"wrote {path}{note}", flush=True)
                written.append(path)
    return written


async def measure_level(
    cfg: BenchmarkConfig,
    server: RunningServer,
    requests: list[Request],
    concurrency: int,
    raw_dir: Path,
) -> list[dict[str, Any]]:
    """Warm up, then ``cfg.runs`` measured runs; returns the metrics of each run."""
    send_options = {"extra_body": cfg.extra_body}
    if cfg.warmup:
        warmup = warmup_set(cfg.warmup, server, requests, concurrency)
        await send_requests(
            server.base_url, cfg.server.model, warmup, max_concurrency=concurrency, **send_options
        )

    per_run = []
    for run in range(cfg.runs):
        await reset_prefix_cache(server)
        with PeakGpuMemory() as gpu:
            timings, wall_time = await send_requests(
                server.base_url,
                cfg.server.model,
                requests,
                max_concurrency=concurrency,
                request_rate=cfg.request_rate,
                seed=cfg.seed + run,
                **send_options,
            )
        metrics = run_metrics(timings, wall_time, cfg.slo_ttft_ms, cfg.slo_tpot_ms)
        metrics["peak_vram_gb"] = gpu.peak_gb
        per_run.append(metrics)
        lines = "".join(json.dumps(t.to_dict()) + "\n" for t in timings)
        (raw_dir / f"c{concurrency}_run{run}.jsonl").write_text(lines)
        print(
            f"[{cfg.server.engine}-{cfg.server.config} c={concurrency} run {run + 1}/{cfg.runs}]"
            f" ttft_p50={metrics['ttft_ms']['p50']:.1f}ms"
            f" tpot_mean={metrics['tpot_ms']['mean']:.2f}ms"
            f" out_tok/s={metrics['output_tokens_per_s']:.1f} failed={metrics['failed']}",
            flush=True,
        )
    return per_run


def warmup_set(
    minimum: int, server: RunningServer, requests: list[Request], concurrency: int
) -> list[Request]:
    """Requests that warm the server up at the shapes about to be measured.

    At least one per concurrency slot, so the first measured run does not pay for kernel
    compilation or graph capture at that batch size. If the engine's prefix cache is reset
    before every run, the warmup reuses the workload's own requests (same prompt lengths),
    sent as independent requests; otherwise synthetic ones, which share no prefix with it.
    """
    n = max(minimum, concurrency)
    if server.spec.cache_reset_endpoint is None:
        return warmup_requests(n)
    pool = requests[:n]
    return [dataclasses.replace(r, id=f"warmup-{r.id}", session_id=None) for r in pool]


def build_result(
    cfg: BenchmarkConfig,
    server: RunningServer,
    workload: str,
    requests: list[Request],
    concurrency: int,
    per_run: list[dict[str, Any]],
    git: environment.GitState,
) -> dict[str, Any]:
    """The result dict of one concurrency level (``bench/results/SCHEMA.md``)."""
    mean, stddev = result_file.aggregate_runs(per_run)
    poisson = math.isfinite(cfg.request_rate)
    result = {
        "engine": cfg.server.engine,
        "engine_version": server.version,
        "engine_config": {
            **server.launch,
            "extra_body": cfg.extra_body,
            "cache_reset_between_runs": server.spec.cache_reset_endpoint,
        },
        "vinfer_commit": git.commit,
        "dirty": git.dirty,
        "model": cfg.server.model,
        "model_revision": cfg.server.revision,
        "workload": workload,
        "concurrency": concurrency,
        "arrival": "poisson" if poisson else "closed-loop",
        "request_rate": cfg.request_rate if poisson else None,
        "seed": cfg.seed,
        "runs": cfg.runs,
        "num_requests": len(requests),
        "num_requests_rule": cfg.num_requests,  # "auto", a fixed int, or None (all)
        "warmup": cfg.warmup,
        "command": cfg.command,
        "hardware": environment.hardware(),
        "software": environment.software(),
        "slo": {"ttft_ms": cfg.slo_ttft_ms, "tpot_ms": cfg.slo_tpot_ms},
        "metrics": mean,
        "stddev": stddev,
        "per_run": per_run,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    }
    problems = result_file.validate(result)
    if cfg.server.engine in ("fake", "external"):
        problems.append(f"engine {cfg.server.engine} is not an official benchmark target")
    result["problems"] = problems
    return result

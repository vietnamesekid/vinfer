"""One result file per (engine, workload, concurrency): aggregate runs, validate, write.

Format: ``bench/results/SCHEMA.md``. A result that fails ``validate`` is still written, but to
``bench/results/raw/unofficial/`` with its problems listed, never next to official results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_KEYS = (
    "engine", "engine_version", "engine_config", "vinfer_commit", "model", "model_revision",
    "workload", "concurrency", "arrival", "request_rate", "seed", "runs", "num_requests",
    "command", "hardware", "software", "slo", "metrics", "stddev", "per_run", "timestamp",
)  # fmt: skip
REQUIRED_METRICS = {
    "ttft_ms": ("p50", "p95", "p99"),
    "tpot_ms": ("mean", "p50", "p99"),
    "output_tokens_per_s": (),
    "goodput_req_per_s": (),
    "peak_vram_gb": (),
    "prefix_cache_hit_rate": (),
}
REQUIRED_HARDWARE = ("provider", "instance_type", "dedicated", "gpu", "num_gpus", "driver", "cuda")


def _combine(values: list[Any], stat: str) -> Any:
    """Mean or sample stddev of scalars, recursively for dicts with the same keys."""
    if isinstance(values[0], dict):
        return {k: _combine([v[k] for v in values], stat) for k in values[0]}
    if any(v is None for v in values):
        return None
    arr = np.asarray(values, dtype=float)
    if stat == "mean":
        return float(arr.mean())
    return float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def aggregate_runs(per_run: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(mean, stddev)`` of per-run metrics; stddev is the sample stddev across runs."""
    if not per_run:
        raise ValueError("no runs to aggregate")
    return _combine(per_run, "mean"), _combine(per_run, "std")


def validate(result: dict[str, Any]) -> list[str]:
    """Problems that make a result unpublishable (docs/benchmarking.md); empty if none."""
    missing = [f"missing key: {k}" for k in REQUIRED_KEYS if k not in result]
    if missing:
        return missing
    problems = []
    metrics = result["metrics"]
    for name, stats in REQUIRED_METRICS.items():
        if name not in metrics:
            problems.append(f"missing metric: {name}")
        else:
            problems += [f"missing metric: {name}.{s}" for s in stats if s not in metrics[name]]
    problems += [f"missing hardware.{k}" for k in REQUIRED_HARDWARE if k not in result["hardware"]]
    if result["engine_version"] in (None, "", "unknown"):
        problems.append("engine_version unknown (baseline venv missing?)")
    if not result["model_revision"]:
        problems.append("model_revision missing (pin it in bench/models.lock)")
    if not result["vinfer_commit"]:
        problems.append("vinfer_commit is empty")
    if result.get("dirty"):
        problems.append("run made from a dirty git tree")
    if result["runs"] < 3:
        problems.append(f"runs={result['runs']} < 3 (docs/benchmarking.md rule 4)")
    if not result["hardware"].get("dedicated"):
        problems.append("hardware.dedicated is not true (bench/HARDWARE.md)")
    if metrics.get("failed"):
        problems.append(f"{metrics['failed']} failed requests per run on average")
    ttft = metrics.get("ttft_ms", {}).get("p50")
    if ttft is None or math.isnan(ttft):
        problems.append("ttft_ms.p50 is not a number")
    return problems


def file_name(stamp: str, result: dict[str, Any]) -> str:
    """``<stamp>_<engine>-<config>_<org--model>_<workload>_c<n>.json``. The stamp is the run's
    start time to the millisecond, so two runs never share a name."""
    model = result["model"].replace("/", "--")
    engine = f"{result['engine']}-{result['engine_config']['config']}"
    return f"{stamp}_{engine}_{model}_{result['workload']}_c{result['concurrency']}.json"


def write(result: dict[str, Any], out_dir: Path, stamp: str) -> Path:
    """Write to ``out_dir`` if ``result["problems"]`` is empty, else to
    ``out_dir/raw/unofficial``. Never overwrites an existing file."""
    dest = out_dir / "raw" / "unofficial" if result["problems"] else out_dir
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / file_name(stamp, result)
    with path.open("x") as f:  # "x" fails instead of replacing an existing result
        json.dump(_nan_to_none(result), f, indent=2, allow_nan=False)
        f.write("\n")
    return path


def _nan_to_none(value: Any) -> Any:
    """NaN (e.g. TPOT when every output had one token) becomes null in the JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value

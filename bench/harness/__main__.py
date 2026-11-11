"""Command line: ``python -m bench.harness <command> [options]``.

Commands:

- ``run``: measure one engine on workloads over a concurrency sweep (``benchmark.py``).
- ``validate``: compare our harness with engines' own benchmark tools (``validation.py``).
- ``fake-server``: start the fake server, to try the harness without a GPU.

Examples::

    python -m bench.harness run --engine fake --workload W1-v1 --num-requests 32 \\
        --concurrency 1 8 --allow-dirty
    python -m bench.harness run --engine vllm --config tuned --model Qwen/Qwen3.5-2B \\
        --workload W1-v1 W2-v1 W3-v1 W4-v1
    python -m bench.harness validate --model Qwen/Qwen3.5-2B --benchmarks vllm sglang
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shlex
import sys
from pathlib import Path

from bench.harness import environment
from bench.harness.benchmark import run_benchmark
from bench.harness.config import (
    RESULTS_DIR,
    BenchmarkConfig,
    HarnessError,
    ServerConfig,
    pinned_revision,
)
from bench.harness.engine_benchmarks import ENGINE_BENCHMARKS
from bench.harness.engines import ENGINES
from bench.harness.fake_server import serve_forever
from bench.harness.validation import ValidationConfig, validate_harness


def _add_server_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="fake-model", help="HF model id")
    p.add_argument("--revision", help="HF revision (default: the pin in bench/models.lock)")
    p.add_argument("--config", default="default", help="baseline launch config, e.g. tuned")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=float, default=1800, help="seconds")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.harness",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="measure one engine on one or more workloads")
    run.add_argument("--engine", required=True, choices=list(ENGINES))
    _add_server_options(run)
    run.add_argument("--base-url", help="server URL, for --engine vinfer / external")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--workload", nargs="+", help="e.g. W1-v1 W4-v1 (directories in bench/workloads/)"
    )
    source.add_argument("--workload-file", type=Path, help="an explicit requests.jsonl")
    run.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 128])
    run.add_argument("--runs", type=int, default=3)
    run.add_argument(
        "--num-requests",
        type=lambda v: v if v in ("auto", "all") else int(v),
        default="auto",
        help="per run: N, 'all', or 'auto' = clamp(8 x concurrency, 64, 512) (default)",
    )
    run.add_argument(
        "--request-rate",
        type=float,
        default=math.inf,
        help="Poisson rate in req/s (default inf: closed loop, capped by concurrency)",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--warmup",
        type=int,
        default=4,
        help="minimum warmup requests per level (at least one per slot)",
    )
    run.add_argument("--slo-ttft-ms", type=float)
    run.add_argument("--slo-tpot-ms", type=float)
    run.add_argument("--extra-body", type=json.loads, help="JSON merged into every request")
    run.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    run.add_argument("--allow-dirty", action="store_true", help="dev only: result is unofficial")

    validate = commands.add_parser("validate", help="compare with engines' own benchmark tools")
    _add_server_options(validate)
    validate.add_argument(
        "--benchmarks", nargs="+", choices=list(ENGINE_BENCHMARKS), default=list(ENGINE_BENCHMARKS)
    )
    validate.add_argument("--workload", default="W1-v1")
    validate.add_argument("--num-requests", type=int, default=200)
    validate.add_argument("--output-len", type=int, default=256)
    validate.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    validate.add_argument("--runs", type=int, default=3)
    validate.add_argument("--out", type=Path, default=RESULTS_DIR / "raw" / "p06-validation.json")

    fake = commands.add_parser("fake-server", help="start the fake server (no GPU needed)")
    fake.add_argument("--port", type=int, default=8001)
    fake.add_argument("--prefill-s", type=float, default=0.02)
    fake.add_argument("--decode-s", type=float, default=0.005)
    return parser


def benchmark_config(a: argparse.Namespace, command: str) -> BenchmarkConfig:
    server = ServerConfig(
        engine=a.engine,
        model=a.model,
        config=a.config,
        revision=a.revision or pinned_revision(a.model),
        port=a.port,
        base_url=a.base_url,
        startup_timeout_s=a.startup_timeout,
    )
    return BenchmarkConfig(
        server=server,
        workloads=tuple(a.workload or ()),
        workload_file=a.workload_file,
        concurrency=tuple(a.concurrency),
        runs=a.runs,
        num_requests=None if a.num_requests == "all" else a.num_requests,
        request_rate=a.request_rate,
        seed=a.seed,
        warmup=a.warmup,
        slo_ttft_ms=a.slo_ttft_ms,
        slo_tpot_ms=a.slo_tpot_ms,
        extra_body=a.extra_body,
        out_dir=a.out_dir,
        allow_dirty=a.allow_dirty,
        command=command,
    )


def validation_config(a: argparse.Namespace) -> ValidationConfig:
    return ValidationConfig(
        model=a.model,
        revision=a.revision or pinned_revision(a.model),
        benchmarks=tuple(a.benchmarks),
        server_config=a.config,
        workload=a.workload,
        num_requests=a.num_requests,
        output_len=a.output_len,
        concurrency=tuple(a.concurrency),
        runs=a.runs,
        port=a.port,
        startup_timeout_s=a.startup_timeout,
        out=a.out,
    )


def main(argv: list[str] | None = None) -> list[Path] | None:
    """Parse ``argv`` and run the command. Returns the result files of ``run``."""
    argv = sys.argv[1:] if argv is None else argv
    environment.load_env_file()
    a = _parser().parse_args(argv)
    command = shlex.join([sys.executable, "-m", "bench.harness", *argv])
    try:
        if a.command == "run":
            return asyncio.run(run_benchmark(benchmark_config(a, command)))
        if a.command == "validate":
            cfg = validation_config(a)
            asyncio.run(validate_harness(cfg))
            print(f"\nwrote {cfg.out}")
        elif a.command == "fake-server":
            asyncio.run(serve_forever(a.port, a.prefill_s, a.decode_s))
    except HarnessError as e:
        sys.exit(f"error: {e}")
    return None


if __name__ == "__main__":
    main()

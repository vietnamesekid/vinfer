"""The ``validate`` command: compare our harness with engines' own benchmark tools.

For each engine benchmark: start that engine's baseline server, then at each concurrency warm
up, and alternate runs of their tool and ours (who goes first alternates too) with a cache
reset before each run, so machine drift and first-run costs hit both sides equally. Both
sides send the same prompts (single-turn requests of a workload) with the same fixed output
length, ``ignore_eos``, a closed loop, and the same concurrency cap.

The output is mean ± stddev per metric and the relative difference. The acceptable tolerance
is a human decision recorded in ``docs/benchmarking.md``; this command only measures.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.harness import environment
from bench.harness.config import RESULTS_DIR, HarnessError, ServerConfig, pinned_revision
from bench.harness.engine_benchmarks import (
    ENGINE_BENCHMARKS,
    FLAT_KEYS,
    EngineBenchmark,
    EngineBenchmarkRun,
)
from bench.harness.engines import RunningServer, reset_prefix_cache, start_server
from bench.harness.load_generator import send_requests
from bench.harness.metrics import run_metrics
from bench.harness.result_file import aggregate_runs
from bench.harness.workload import Request, load_workload


@dataclass(frozen=True)
class ValidationConfig:
    model: str
    revision: str | None = None
    benchmarks: tuple[str, ...] = tuple(ENGINE_BENCHMARKS)
    server_config: str = "default"  # baseline config the servers run
    workload: str = "W1-v1"
    num_requests: int = 200
    output_len: int = 256
    concurrency: tuple[int, ...] = (1, 8, 32)
    runs: int = 3
    port: int = 8000
    startup_timeout_s: float = 1800.0
    out: Path = RESULTS_DIR / "raw" / "p06-validation.json"


# SGLang's custom-dataset loader silently drops prompts under 2 tokens (datasets/custom.py,
# v0.5.23); dropping them on every side keeps both sides on the same prompts.
MIN_PROMPT_TOKENS = 2


def validation_requests(
    workload: str,
    n: int,
    output_len: int,
    tokenizer: str | None = None,
    revision: str | None = None,
) -> list[Request]:
    """The first ``n`` single-turn requests of ``workload``, all with ``output_len`` tokens.

    Single-turn only, and one fixed output length, because not every benchmark tool supports
    multi-turn requests or a per-request output length. With ``tokenizer`` (a model id),
    prompts under ``MIN_PROMPT_TOKENS`` raw tokens (no chat template) are skipped.
    """
    count_tokens = _token_counter(tokenizer, revision) if tokenizer else None
    _, requests = load_workload(workload)
    single = [
        r
        for r in requests
        if r.is_single_turn
        and (count_tokens is None or count_tokens(r.messages[0]["content"]) >= MIN_PROMPT_TOKENS)
    ]
    if len(single) < n:
        raise HarnessError(f"{workload} has only {len(single)} single-turn requests, need {n}")
    return [dataclasses.replace(r, max_tokens=output_len) for r in single[:n]]


def compare(ours: list[dict], theirs: list[dict]) -> dict[str, dict[str, float | None]]:
    """Per metric: mean and stddev of both sides, and the relative difference ours vs theirs."""
    ours_mean, ours_std = aggregate_runs(ours)
    theirs_mean, theirs_std = aggregate_runs(theirs)

    def get(metrics: dict[str, Any], section: str, key: str | None) -> float:
        return metrics[section] if key is None else metrics[section][key]

    rows = {}
    for section, key in FLAT_KEYS.values():
        o, t = get(ours_mean, section, key), get(theirs_mean, section, key)
        rows[f"{section}.{key}" if key else section] = {
            "ours": o,
            "ours_std": get(ours_std, section, key),
            "theirs": t,
            "theirs_std": get(theirs_std, section, key),
            "rel_diff": (o - t) / t if t else None,
        }
    return rows


async def validate_harness(cfg: ValidationConfig) -> dict[str, Any]:
    """Run every benchmark in ``cfg.benchmarks``. The report is rewritten to ``cfg.out`` after
    every concurrency level, so a failure later on never loses finished levels. A benchmark
    that fails is recorded with its error and the next one still runs; raises at the end if
    any failed."""
    requests = validation_requests(
        cfg.workload, cfg.num_requests, cfg.output_len, cfg.model, cfg.revision
    )
    work_dir = cfg.out.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": cfg.model,
        "model_revision": cfg.revision,
        "workload": cfg.workload,
        "num_requests": len(requests),
        "output_len": cfg.output_len,
        "runs": cfg.runs,
        "commit": environment.git_state().commit,
        "hardware": environment.hardware(),
        "benchmarks": {},
    }

    def save() -> None:
        cfg.out.write_text(json.dumps(report, indent=2))

    failed = []
    for name in cfg.benchmarks:
        entry: dict[str, Any] = {"server_config": cfg.server_config, "levels": {}}
        report["benchmarks"][name] = entry
        try:
            await _validate_benchmark(cfg, ENGINE_BENCHMARKS[name], requests, work_dir, entry, save)
        except HarnessError as e:
            entry["error"] = str(e)
            failed.append(name)
            print(f"\n{name}: FAILED: {e}", flush=True)
        save()
    if failed:
        raise HarnessError(f"validation failed for {', '.join(failed)} (details in {cfg.out})")
    return report


def _token_counter(model: str, revision: str | None) -> Callable[[str], int]:
    """Raw token count of a prompt with the model's tokenizer, as the tools count it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision or pinned_revision(model))
    return lambda text: len(tokenizer.encode(text))


async def _validate_benchmark(
    cfg: ValidationConfig,
    benchmark: EngineBenchmark,
    requests: list[Request],
    work_dir: Path,
    entry: dict[str, Any],
    save: Callable[[], None],
) -> None:
    """Fill ``entry`` level by level, calling ``save`` after each one."""
    dataset = work_dir / f"{benchmark.name}.dataset.jsonl"
    benchmark.write_dataset([r.messages[0]["content"] for r in requests], dataset)
    # Ours sends the prompts in the order the tool does: at a concurrency above 1 the order
    # decides which prompts are prefilled together, which moves TTFT.
    requests = [requests[i] for i in benchmark.send_order(len(requests))]
    server_cfg = ServerConfig(
        engine=benchmark.engine,
        model=cfg.model,
        config=cfg.server_config,
        revision=cfg.revision,
        port=cfg.port,
        startup_timeout_s=cfg.startup_timeout_s,
    )
    async with start_server(server_cfg, work_dir / f"{benchmark.name}.server.log") as server:
        entry["engine_version"] = server.version
        for concurrency in cfg.concurrency:
            # Warm up at this batch shape, so neither side pays first-run costs (compilation,
            # graph capture, allocator growth) inside a measured run.
            await _run_ours(server, cfg, requests[: max(4, concurrency)], concurrency)
            ours, theirs = [], []
            for run in range(cfg.runs):
                benchmark_run = EngineBenchmarkRun(
                    base_url=server.base_url,
                    model=cfg.model,
                    dataset=dataset,
                    num_requests=len(requests),
                    output_len=cfg.output_len,
                    concurrency=concurrency,
                    result=work_dir / f"{benchmark.name}.c{concurrency}.run{run}.json",
                )
                # Alternate who goes first, so drift within a level hits both sides equally.
                for side in ("theirs", "ours") if run % 2 == 0 else ("ours", "theirs"):
                    await reset_prefix_cache(server)
                    if side == "theirs":
                        theirs.append(await _run_benchmark(benchmark, benchmark_run))
                    else:
                        ours.append(
                            await _run_ours(
                                server,
                                cfg,
                                requests,
                                concurrency,
                                # Per-request records: mean-only comparison cannot show whether
                                # a gap comes from all requests or from a few stragglers.
                                out=benchmark_run.result.with_suffix(".ours.jsonl"),
                            )
                        )
            entry["levels"][f"c{concurrency}"] = compare(ours, theirs)
            save()
            _print_level(benchmark.name, concurrency, entry["levels"][f"c{concurrency}"])


async def _run_benchmark(benchmark: EngineBenchmark, run: EngineBenchmarkRun) -> dict[str, Any]:
    """Run the engine's tool; its output goes to ``<result>.log``."""
    run.result.unlink(missing_ok=True)  # SGLang appends to its result file
    log_path = run.result.with_suffix(".log")
    with log_path.open("w") as log:
        process = await asyncio.to_thread(
            subprocess.run, benchmark.command(run), stdout=log, stderr=subprocess.STDOUT
        )
    if process.returncode != 0 or not run.result.exists():
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-5:])
        raise HarnessError(
            f"{benchmark.name} benchmark exited with code {process.returncode}; see {log_path}\n"
            f"{tail}"
        )
    return benchmark.read_result(run.result)


async def _run_ours(
    server: RunningServer,
    cfg: ValidationConfig,
    requests: list[Request],
    concurrency: int,
    out: Path | None = None,
) -> dict[str, Any]:
    timings, wall_time = await send_requests(
        server.base_url,
        cfg.model,
        requests,
        max_concurrency=concurrency,
    )
    if out is not None:
        out.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in timings))
    return run_metrics(timings, wall_time)


def _print_level(
    benchmark: str, concurrency: int, rows: dict[str, dict[str, float | None]]
) -> None:
    print(f"\n{benchmark} @ concurrency {concurrency}")
    for name, r in rows.items():
        rel = "n/a" if r["rel_diff"] is None else f"{100 * r['rel_diff']:+.1f}%"
        print(
            f"  {name:20s} ours {r['ours']:10.2f} ± {r['ours_std']:<8.2f}"
            f" {benchmark} {r['theirs']:10.2f} ± {r['theirs_std']:<8.2f} {rel}",
            flush=True,
        )

"""Other engines' own benchmark tools, wrapped so our harness can be checked against them.

Each engine ships its own load generator (``vllm bench serve``, ``sglang.benchmark.serving``).
If our harness reports the same numbers as that tool on the same server and requests, our
metric definitions can be trusted. An ``EngineBenchmark`` adapts one tool:

- ``write_dataset``: write the prompts in the tool's own dataset format,
- ``command``: the exact command line that runs the tool,
- ``read_result``: turn the tool's result file into our metric layout (``metrics.py``),
- ``send_order``: the order in which the tool sends the prompts, so ours sends them in the same
  order (at a concurrency above 1 the order decides which prompts share a batch).

Adding an engine: subclass ``EngineBenchmark`` with flags read from that engine's source at a
pinned version, add the engine to ``engines.ENGINES`` with its launch scripts under
``bench/baselines/<engine>/``, and register the benchmark in ``ENGINE_BENCHMARKS``.
"""

from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from bench.harness.config import REPO

SEED = 0  # passed to every tool with --seed

# Result keys shared by vLLM and SGLang -> (section, key) in our metric layout.
FLAT_KEYS = {
    "median_ttft_ms": ("ttft_ms", "p50"),
    "p99_ttft_ms": ("ttft_ms", "p99"),
    "mean_tpot_ms": ("tpot_ms", "mean"),
    "p99_tpot_ms": ("tpot_ms", "p99"),
    "median_itl_ms": ("itl_ms", "p50"),
    "output_throughput": ("output_tokens_per_s", None),
}


@dataclass(frozen=True)
class EngineBenchmarkRun:
    """One run of an engine's benchmark tool."""

    base_url: str
    model: str
    dataset: Path
    num_requests: int
    output_len: int  # fixed for every request: not every tool supports per-request lengths
    concurrency: int
    result: Path


class EngineBenchmark(ABC):
    name: ClassVar[str]
    engine: ClassVar[str]  # the engine whose server it measures (a key of engines.ENGINES)

    def executable(self, name: str) -> Path:
        """``name`` inside this engine's venv (scripts/setup_baselines.sh)."""
        venvs = Path(os.environ.get("VINFER_VENVS", REPO / ".venvs"))
        return venvs / self.engine / "bin" / name

    @abstractmethod
    def write_dataset(self, prompts: list[str], path: Path) -> None: ...

    @abstractmethod
    def command(self, run: EngineBenchmarkRun) -> list[str]: ...

    @abstractmethod
    def read_result(self, path: Path) -> dict[str, Any]: ...

    def send_order(self, n: int) -> list[int]:
        """Indices of the ``n`` dataset prompts in the order the tool sends them."""
        return list(range(n))


def from_flat(result: dict[str, Any]) -> dict[str, Any]:
    """Flat ``FLAT_KEYS`` result -> our nested layout, e.g. ``{"ttft_ms": {"p50": ...}}``."""
    out: dict[str, Any] = {}
    for flat, (section, key) in FLAT_KEYS.items():
        value = float(result[flat])
        if key is None:
            out[section] = value
        else:
            out.setdefault(section, {})[key] = value
    return out


class VllmBenchServe(EngineBenchmark):
    """``vllm bench serve`` (vLLM v0.33.0, vllm/benchmarks/serve.py)."""

    name = "vllm"
    engine = "vllm"

    def write_dataset(self, prompts: list[str], path: Path) -> None:
        # `custom` dataset: one JSON object per line with a "prompt" (datasets.py, CustomDataset).
        path.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in prompts))

    def command(self, run: EngineBenchmarkRun) -> list[str]:
        return [
            str(self.executable("vllm")), "bench", "serve",
            "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
            "--base-url", run.base_url, "--model", run.model, "--tokenizer", run.model,
            "--dataset-name", "custom", "--dataset-path", str(run.dataset),
            "--skip-chat-template",  # send the prompt as the user message, as we do
            "--disable-shuffle",  # keep the dataset order, which is ours (datasets.py)
            "--custom-output-len", str(run.output_len), "--ignore-eos",
            # Greedy, as we send. Without it the server default applies: Qwen3.5 ships no
            # generation_config.json, so requests run at temperature 1.0 (protocol.py).
            "--temperature", "0",
            "--num-prompts", str(run.num_requests), "--max-concurrency", str(run.concurrency),
            "--request-rate", "inf", "--num-warmups", "0", "--seed", str(SEED),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
            "--save-result", "--result-dir", str(run.result.parent),
            "--result-filename", run.result.name,
        ]  # fmt: skip

    def read_result(self, path: Path) -> dict[str, Any]:
        return from_flat(json.loads(path.read_text()))


class SglangBenchServing(EngineBenchmark):
    """``python -m sglang.benchmark.serving`` (SGLang v0.5.23, sglang/benchmark/serving.py)."""

    name = "sglang"
    engine = "sglang"

    def write_dataset(self, prompts: list[str], path: Path) -> None:
        # `custom` dataset: {"conversations": [user, assistant]} per line; the assistant turn is
        # ignored because --sharegpt-output-len fixes the output length (datasets/custom.py).
        rows = [
            {
                "conversations": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": ""},
                ]
            }
            for p in prompts
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def command(self, run: EngineBenchmarkRun) -> list[str]:
        return [
            str(self.executable("python")), "-m", "sglang.benchmark.serving",
            "--backend", "sglang-oai-chat", "--base-url", run.base_url,
            "--model", run.model, "--tokenizer", run.model,
            "--dataset-name", "custom", "--dataset-path", str(run.dataset),
            "--sharegpt-output-len", str(run.output_len),  # ignore_eos is on by default
            "--num-prompts", str(run.num_requests), "--max-concurrency", str(run.concurrency),
            "--request-rate", "inf", "--warmup-requests", "0", "--seed", str(SEED),
            "--output-file", str(run.result),
        ]  # fmt: skip

    def send_order(self, n: int) -> list[int]:
        # The tool always shuffles: run_benchmark calls random.seed(args.seed), then
        # sample_custom_requests calls random.shuffle(dataset) with no random call in between
        # (serving.py, datasets/custom.py). Its loader also drops prompts under 2 tokens;
        # validation_requests drops those for every tool, so no index shifts.
        order = list(range(n))
        random.Random(SEED).shuffle(order)
        return order

    def read_result(self, path: Path) -> dict[str, Any]:
        # The tool appends one JSON line per run to --output-file; ours is the last line.
        last = path.read_text().strip().splitlines()[-1]
        return from_flat(json.loads(last))


ENGINE_BENCHMARKS: dict[str, EngineBenchmark] = {
    c.name: c for c in (VllmBenchServe(), SglangBenchServing())
}

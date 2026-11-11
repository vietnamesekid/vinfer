"""Paths, the harness error type, and the typed configs every command is built from."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "bench" / "results"
BASELINES_DIR = REPO / "bench" / "baselines"
WORKLOADS_DIR = REPO / "bench" / "workloads"
MODELS_LOCK = REPO / "bench" / "models.lock"
ENV_FILE = REPO / ".env.bench"


class HarnessError(RuntimeError):
    """The harness cannot do what was asked. The CLI prints it and exits non-zero."""


def pinned_revision(model: str) -> str | None:
    """The revision pinned for ``model`` in ``bench/models.lock``, if any."""
    if not MODELS_LOCK.exists():
        return None
    for line in MODELS_LOCK.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == model:
            return parts[1]
    return None


@dataclass(frozen=True)
class ServerConfig:
    """Which server to measure, and how to start it."""

    engine: str  # a key of engines.ENGINES
    model: str
    config: str = "default"  # launch script bench/baselines/<engine>/configs/<config>.sh
    revision: str | None = None
    port: int = 8000
    base_url: str | None = None  # for servers we do not start ourselves
    startup_timeout_s: float = 1800.0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Everything the ``run`` command needs; recorded in every result file."""

    server: ServerConfig
    workloads: tuple[str, ...] = ()  # names in bench/workloads/, measured on one server launch
    workload_file: Path | None = None  # or one explicit jsonl file
    concurrency: tuple[int, ...] = (1, 8, 32, 128)
    runs: int = 3
    num_requests: int | str | None = "auto"  # per run: int, "auto" (workload.py), None = all
    request_rate: float = math.inf  # Poisson rate in req/s; inf = all requests at t=0
    seed: int = 0
    warmup: int = 4  # synthetic warmup requests per concurrency level
    slo_ttft_ms: float | None = None
    slo_tpot_ms: float | None = None
    extra_body: dict[str, Any] | None = None  # merged into every request body
    out_dir: Path = RESULTS_DIR
    allow_dirty: bool = False
    command: str = ""  # the command line, for the record

    def __post_init__(self) -> None:
        if bool(self.workloads) == (self.workload_file is not None):
            raise HarnessError("give workload names or one workload file, not both or neither")
        if self.runs < 1 or not self.concurrency:
            raise HarnessError("need at least one run and one concurrency level")

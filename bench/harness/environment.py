"""What a result must record about the machine: commit, hardware, software.

Facts the machine cannot discover itself come from environment variables, which
``scripts/setup.sh`` writes to ``.env.bench`` (``KEY=VALUE`` lines) on a rented machine:

- ``VINFER_PROVIDER`` (e.g. ``runpod``) and ``VINFER_INSTANCE_TYPE``,
- ``VINFER_DEDICATED=1`` only for dedicated (non-spot, non-shared) instances.

``load_env_file`` reads that file; the Makefile includes it too.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from bench.harness.config import ENV_FILE, REPO


@dataclass(frozen=True)
class GitState:
    commit: str | None
    dirty: bool  # uncommitted changes outside bench/results/


def load_env_file(path: Path = ENV_FILE) -> None:
    """Use ``KEY=VALUE`` lines of ``path`` as environment defaults (set variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def _output(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> GitState:
    commit = _output(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    changes = _output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", ".", ":(exclude)bench/results"]
    )
    return GitState(commit=commit, dirty=bool(changes))


def hardware() -> dict[str, Any]:
    query = "--query-gpu=name,driver_version,memory.total"
    rows = _output(["nvidia-smi", query, "--format=csv,noheader"])
    gpus = [row.split(", ") for row in rows.splitlines()] if rows else []
    smi = _output(["nvidia-smi"]) or ""
    cuda = smi.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in smi else None
    first = gpus[0] if gpus else [None, None, None]
    return {
        "provider": os.environ.get("VINFER_PROVIDER"),
        "instance_type": os.environ.get("VINFER_INSTANCE_TYPE"),
        "dedicated": os.environ.get("VINFER_DEDICATED") == "1",
        "gpu": first[0],
        "gpu_memory": first[2],
        "num_gpus": len(gpus),
        "driver": first[1],
        "cuda": cuda,
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "host": platform.node(),
    }


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def software() -> dict[str, Any]:
    """Client-side software. Engine versions come from the engine's own venv (engines.py)."""
    return {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "httpx": _version("httpx"),
        "os": platform.platform(),
    }

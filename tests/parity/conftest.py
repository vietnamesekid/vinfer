"""Fixtures for GPU parity tests. Configure with environment variables (``make parity`` sets
them; ``.env.bench`` on a rented machine provides the cache location):

- ``VINFER_PARITY_MODEL`` (required): HF model id.
- ``VINFER_PARITY_REVISION``: HF revision; defaults to the pin in ``bench/models.lock``.
- ``VINFER_PARITY_CACHE``: where reference summaries are cached (default ``weights/parity``).
- ``VINFER_PARITY_DTYPE``: reference dtype (default bfloat16).
- ``VINFER_PARITY_KERNELS``: vinfer kernel backends to check, e.g.
  ``attention_backend=reference,gdn_backend=reference`` (default: the reference ones).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.harness.config import pinned_revision
from bench.harness.environment import load_env_file
from vinfer.config import KernelConfig


@pytest.fixture(scope="session")
def parity_env() -> dict:
    load_env_file()
    model = os.environ.get("VINFER_PARITY_MODEL")
    if not model:
        pytest.skip("set VINFER_PARITY_MODEL (make parity MODEL=...)")
    revision = os.environ.get("VINFER_PARITY_REVISION") or pinned_revision(model)
    if not revision:
        pytest.fail(f"no revision for {model}: pin it in bench/models.lock or set REVISION")
    return {
        "model": model,
        "revision": revision,
        "dtype": os.environ.get("VINFER_PARITY_DTYPE", "bfloat16"),
        "cache": Path(os.environ.get("VINFER_PARITY_CACHE", "weights/parity")),
        "kernels": KernelConfig.from_string(os.environ.get("VINFER_PARITY_KERNELS")),
    }

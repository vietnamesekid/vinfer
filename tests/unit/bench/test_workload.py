"""Loading and selecting workload requests (bench/harness/workload.py)."""

from __future__ import annotations

import pytest

from bench.harness.config import HarnessError
from bench.harness.workload import load_workload, requests_per_run


def test_auto_request_count_scales_with_concurrency_within_bounds():
    assert [requests_per_run("auto", c) for c in (1, 8, 32, 128)] == [64, 64, 256, 512]
    assert requests_per_run(None, 8) is None and requests_per_run(10, 8) == 10


def test_load_errors_are_exceptions_not_exits():
    with pytest.raises(HarnessError):
        load_workload(None, None)
    with pytest.raises(HarnessError, match="make workloads"):
        load_workload("W9-v9")

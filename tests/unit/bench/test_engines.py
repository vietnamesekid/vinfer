"""Server lifecycle and prefix-cache resets (bench/harness/engines.py)."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from bench.harness import engines
from bench.harness.config import HarnessError


def test_launch_refuses_a_busy_port():
    """A stale server still holding the port must not be measured under a new name."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        with pytest.raises(HarnessError, match="already in use"):
            engines._check_port_free(s.getsockname()[1])


def test_reset_counts_success_false_as_failure():
    """vLLM answers HTTP 200 with {"success": false} when it could not reset the cache."""
    assert not engines._reset_succeeded(httpx.Response(200, json={"success": False}))
    assert engines._reset_succeeded(httpx.Response(200, json={"success": True}))
    assert engines._reset_succeeded(httpx.Response(200, text="Cache flushed."))  # SGLang


def test_reset_retries_on_connection_errors_then_raises():
    server = engines.RunningServer("http://127.0.0.1:9", engines.ENGINES["vllm"], "v")
    with pytest.raises(HarnessError, match="failed 2 times"):
        asyncio.run(engines.reset_prefix_cache(server, attempts=2, retry_delay_s=0))


def test_sglang_flush_waits_for_running_requests():
    assert "timeout=" in engines.ENGINES["sglang"].cache_reset_endpoint


def test_stop_waits_for_every_process_of_the_server(tmp_path):
    """Children (engine cores) must be gone before the next server needs the port and GPU."""
    ready = tmp_path / "child-ready"
    child = (
        "import pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    leader = subprocess.Popen(
        ["bash", "-c", f'"{sys.executable}" -c "{child}" & wait'], start_new_session=True
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.exists()

    started = time.monotonic()
    engines._stop(leader, timeout_s=1)
    assert time.monotonic() - started >= 1  # SIGTERM was ignored: escalated to SIGKILL
    with pytest.raises(ProcessLookupError):
        os.killpg(leader.pid, 0)

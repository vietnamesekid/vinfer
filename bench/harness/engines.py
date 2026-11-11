"""The engines we can measure, and starting / stopping their servers.

Three kinds of server:

- baselines (``vllm``, ``sglang``): started from
  ``bench/baselines/<engine>/configs/<config>.sh``
  in the engine's own venv, stopped when the measurement ends;
- servers someone else started (``vinfer``, ``external``): reached through ``base_url``;
- ``fake``: the in-process fake server, for CPU tests.

Adding an engine: add an ``EngineSpec`` to ``ENGINES`` and, if we launch it, a directory
``bench/baselines/<engine>/`` with ``configs/<config>.sh`` and ``version.sh``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import signal
import socket
import subprocess
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bench.harness.config import BASELINES_DIR, REPO, HarnessError, ServerConfig
from bench.harness.environment import git_state
from bench.harness.fake_server import FakeServer


@dataclass(frozen=True)
class EngineSpec:
    launched: bool  # True: we start it from bench/baselines/<engine>/configs/<config>.sh
    # POST endpoint that empties the prefix cache. Called before every measured run so that
    # no run starts with a warm cache. None: the engine has no cross-request cache to reset.
    cache_reset_endpoint: str | None = None


ENGINES: dict[str, EngineSpec] = {
    # Needs VLLM_SERVER_DEV_MODE=1 (set in bench/baselines/vllm/*.sh).
    "vllm": EngineSpec(launched=True, cache_reset_endpoint="/reset_prefix_cache"),
    # timeout: wait up to 60 s for running requests to drain instead of refusing at once.
    "sglang": EngineSpec(launched=True, cache_reset_endpoint="/flush_cache?timeout=60"),
    "vinfer": EngineSpec(launched=False),
    "external": EngineSpec(launched=False),
    "fake": EngineSpec(launched=False, cache_reset_endpoint="/reset_prefix_cache"),
}


@dataclass
class RunningServer:
    base_url: str
    spec: EngineSpec
    version: str  # engine version, recorded in results
    launch: dict[str, Any] = field(default_factory=dict)  # how it was started, for the record


@contextlib.asynccontextmanager
async def start_server(cfg: ServerConfig, log_path: Path) -> AsyncGenerator[RunningServer, None]:
    """Start (or reach) the server described by ``cfg``; stop it on exit if we started it."""
    spec = ENGINES[cfg.engine]
    if cfg.engine == "fake":
        fake = FakeServer()
        port = await fake.start()
        launch = {"config": cfg.config, "prefill_s": fake.prefill_s, "decode_s": fake.decode_s}
        try:
            yield RunningServer(f"http://127.0.0.1:{port}", spec, "fake", launch)
        finally:
            await fake.stop()
    elif spec.launched:
        async with _launch_baseline(cfg, spec, log_path) as server:
            yield server
    else:
        if not cfg.base_url:
            raise HarnessError(f"--base-url is required for engine {cfg.engine}")
        base_url = cfg.base_url.rstrip("/")
        await _wait_until_healthy(base_url, None, cfg.startup_timeout_s)
        version = (git_state().commit or "unknown") if cfg.engine == "vinfer" else cfg.engine
        yield RunningServer(base_url, spec, version, {"config": cfg.config})


async def reset_prefix_cache(
    server: RunningServer, attempts: int = 10, retry_delay_s: float = 3.0
) -> None:
    """Empty the server's prefix cache, retrying while the engine refuses (e.g. right after a
    run, while requests are still finishing).

    Success means HTTP 200 *and* no ``"success": false`` in a JSON body: vLLM answers 200 with
    ``{"success": false}`` when it could not reset (vllm/entrypoints/serve/dev/cache/
    api_router.py), SGLang answers 400.
    """
    endpoint = server.spec.cache_reset_endpoint
    if endpoint is None:
        return
    detail = ""
    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(attempts):
            try:
                response = await client.post(server.base_url + endpoint)
            except httpx.HTTPError as e:  # busy or restarting server: retry like a refusal
                detail = f"{type(e).__name__}: {e}"
            else:
                if response.status_code == 200 and _reset_succeeded(response):
                    return
                detail = f"HTTP {response.status_code} {response.text[:200]}"
            await asyncio.sleep(retry_delay_s)
    raise HarnessError(f"cache reset {endpoint} failed {attempts} times: {detail}")


def _reset_succeeded(response: httpx.Response) -> bool:
    try:
        body = response.json()
    except ValueError:
        return True  # plain-text bodies (SGLang) carry no success flag
    return not (isinstance(body, dict) and body.get("success") is False)


# ----------------------------------------------------------------------------- baselines


@contextlib.asynccontextmanager
async def _launch_baseline(
    cfg: ServerConfig, spec: EngineSpec, log_path: Path
) -> AsyncGenerator[RunningServer, None]:
    script = BASELINES_DIR / cfg.engine / "configs" / f"{cfg.config}.sh"
    if not script.exists():
        raise HarnessError(f"no launch script {script}")
    _check_port_free(cfg.port)
    script_env = dict(
        os.environ,
        VINFER_MODEL=cfg.model,
        VINFER_PORT=str(cfg.port),
        VINFER_MODEL_REVISION=cfg.revision or "",
    )
    # Record every script that shapes the server: the config and the engine's harness settings.
    harness = BASELINES_DIR / cfg.engine / "harness.sh"
    text = script.read_text()
    harness_text = harness.read_text() if harness.exists() else ""
    launch = {
        "config": cfg.config,
        "launch_script": str(script.relative_to(REPO)),
        "launch_script_sha256": hashlib.sha256((text + harness_text).encode()).hexdigest(),
        "launch_script_text": text,
        "harness_script_text": harness_text,
    }
    version = _engine_version(cfg.engine, script_env)
    base_url = f"http://127.0.0.1:{cfg.port}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            ["bash", str(script)],
            env=script_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so we can stop all its children
        )
        try:
            await _wait_until_healthy(base_url, process, cfg.startup_timeout_s)
            yield RunningServer(base_url, spec, version, launch)
        finally:
            _stop(process)


def _engine_version(engine: str, script_env: dict[str, str]) -> str:
    script = BASELINES_DIR / engine / "version.sh"
    result = subprocess.run(["bash", str(script)], env=script_env, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def _check_port_free(port: int) -> None:
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise HarnessError(f"port {port} is already in use: stop the old server first")


async def _wait_until_healthy(
    base_url: str, process: subprocess.Popen | None, timeout_s: float
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    async with httpx.AsyncClient(timeout=5) as client:
        while loop.time() < deadline:
            if process is not None and process.poll() is not None:
                raise HarnessError(f"server exited with code {process.returncode} at startup")
            with contextlib.suppress(httpx.HTTPError):
                if (await client.get(base_url + "/health")).status_code == 200:
                    return
            await asyncio.sleep(2)
    raise HarnessError(f"server at {base_url} not healthy after {timeout_s:.0f}s")


def _stop(process: subprocess.Popen, timeout_s: float = 120.0) -> None:
    """Stop the server and every child it spawned (engine cores, schedulers, ...), and return
    only once they are all gone, so the next server finds the port and GPU memory free."""
    _signal_group(process, signal.SIGTERM)
    if not _wait_group_exit(process, timeout_s):
        _signal_group(process, signal.SIGKILL)
        _wait_group_exit(process, 30)


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    # ProcessLookupError: already gone. PermissionError: only zombies left (macOS).
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, sig)


def _wait_group_exit(process: subprocess.Popen, timeout_s: float) -> bool:
    """True once no process of the group is left (the leader is also reaped)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        process.poll()  # reap the leader so it does not linger as a zombie
        try:
            os.killpg(process.pid, 0)  # raises ProcessLookupError once the group is empty
        except ProcessLookupError:
            return True
        except PermissionError:
            pass  # macOS: members are exiting zombies, not yet reaped; keep waiting
        time.sleep(0.2)
    return False

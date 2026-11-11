"""Baseline launch scripts and machine setup, checked without a GPU."""

from __future__ import annotations

import subprocess

import pytest

from bench.harness.config import REPO

LAUNCH_CONFIGS = sorted((REPO / "bench/baselines").glob("*/configs/*.sh"))
CACHE_REPORT_FLAG = {"vllm": "--enable-prompt-tokens-details", "sglang": "--enable-cache-report"}


@pytest.mark.parametrize("script", LAUNCH_CONFIGS, ids=lambda p: f"{p.parts[-3]}-{p.stem}")
def test_launch_configs_pass_the_harness_settings(script, tmp_path):
    """Every config reports cached tokens and, for vLLM, enables the cache-reset endpoint.

    Runs the real script against a fake venv whose executables print what they receive.
    """
    engine = script.parts[-3]
    bin_dir = tmp_path / engine / "bin"
    bin_dir.mkdir(parents=True)
    for exe in ("python", "vllm"):
        fake = bin_dir / exe
        fake.write_text('#!/usr/bin/env bash\necho "DEV_MODE=${VLLM_SERVER_DEV_MODE:-} ARGS=$*"\n')
        fake.chmod(0o755)
    env = {
        "PATH": "/usr/bin:/bin",
        "VINFER_VENVS": str(tmp_path),
        "VINFER_MODEL": "org/m",
        "VINFER_PORT": "8123",
        "VINFER_MODEL_REVISION": "abc",
    }
    out = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert CACHE_REPORT_FLAG[engine] in out.stdout
    assert "--port 8123" in out.stdout and "--revision abc" in out.stdout
    if engine == "vllm":
        assert "DEV_MODE=1" in out.stdout  # needed for /reset_prefix_cache


def test_vllm_configs_use_the_default_frontend():
    """The experimental Rust frontend would make startup depend on an extra binary."""
    for script in (REPO / "bench/baselines/vllm/configs").glob("*.sh"):
        assert "VLLM_USE_RUST_FRONTEND" not in script.read_text(), script


def test_setup_checks_git_inside_the_repo_and_writes_env_first():
    script = (REPO / "scripts/setup.sh").read_text()
    assert script.index('cd "$(dirname "$0")/.."') < script.index("git rev-parse HEAD")
    assert script.index("cat > .env.bench") < script.index("uv sync")


def test_local_weights_directory_is_ignored_by_git():
    """A `weights` symlink must not make the tree dirty (dirty trees give unofficial results)."""
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--no-index", "-q", "weights"], check=False
    )
    assert result.returncode == 0

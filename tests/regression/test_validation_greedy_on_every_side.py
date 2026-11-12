"""``validate`` must have each tool send greedy requests, as our harness does.

Bug: ``vllm bench serve`` (v0.33.0) sends no temperature unless ``--temperature`` is given, so the
server default applied. Qwen3.5-2B ships no generation_config.json, so the tool's requests ran
at temperature 1.0 (full-vocabulary sampling) while ours ran at 0.0.
"""

from __future__ import annotations

from bench.harness.engine_benchmarks import ENGINE_BENCHMARKS
from bench.harness.http_client import build_payload
from bench.harness.workload import Request
from tests.unit.bench.test_validation import RUN


def test_ours_is_greedy():
    payload = build_payload("m", Request("0", [{"role": "user", "content": "hi"}], 4))
    assert payload["temperature"] == 0.0


def test_vllm_tool_is_greedy():
    command = ENGINE_BENCHMARKS["vllm"].command(RUN)
    assert float(command[command.index("--temperature") + 1]) == 0.0


def test_sglang_tool_is_greedy_by_default():
    # sglang.benchmark.serving's chat request sets temperature 0.0 unless the extra request
    # body overrides it; we pass no extra body.
    command = ENGINE_BENCHMARKS["sglang"].command(RUN)
    assert "--temperature" not in command and "--extra-request-body" not in command

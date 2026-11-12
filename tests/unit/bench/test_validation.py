"""Engine-benchmark adapters and the ``validate`` command, on CPU (fake server and tool)."""

from __future__ import annotations

import asyncio
import itertools
import json
import sys
from pathlib import Path

import pytest

from bench.harness import validation
from bench.harness.config import HarnessError
from bench.harness.engine_benchmarks import ENGINE_BENCHMARKS, EngineBenchmark, EngineBenchmarkRun
from bench.harness.workload import Request

RUN = EngineBenchmarkRun("http://h:1", "org/m", Path("/d.jsonl"), 50, 256, 8, Path("/out/r.json"))
FLAT_RESULT = {
    "median_ttft_ms": 10.0,
    "p99_ttft_ms": 30.0,
    "mean_tpot_ms": 5.0,
    "p99_tpot_ms": 7.0,
    "median_itl_ms": 4.0,
    "output_throughput": 900.0,
}


@pytest.mark.parametrize("name", list(ENGINE_BENCHMARKS))
def test_commands_pin_the_comparison_settings(name):
    command = ENGINE_BENCHMARKS[name].command(RUN)
    flags = dict(itertools.pairwise(command))
    assert flags["--max-concurrency"] == "8" and flags["--num-prompts"] == "50"
    assert flags["--request-rate"] == "inf" and flags["--base-url"] == "http://h:1"
    assert "256" in command  # the fixed output length


def test_vllm_benchmark_forces_eos_and_skips_its_chat_template():
    command = ENGINE_BENCHMARKS["vllm"].command(RUN)
    assert "--ignore-eos" in command and "--skip-chat-template" in command


def test_dataset_formats(tmp_path):
    ENGINE_BENCHMARKS["vllm"].write_dataset(["hi"], tmp_path / "v.jsonl")
    assert json.loads((tmp_path / "v.jsonl").read_text()) == {"prompt": "hi"}
    ENGINE_BENCHMARKS["sglang"].write_dataset(["hi"], tmp_path / "s.jsonl")
    turns = json.loads((tmp_path / "s.jsonl").read_text())["conversations"]
    assert turns[0] == {"role": "user", "content": "hi"} and turns[1]["role"] == "assistant"


def test_read_results(tmp_path):
    (tmp_path / "v.json").write_text(json.dumps(FLAT_RESULT))
    got = ENGINE_BENCHMARKS["vllm"].read_result(tmp_path / "v.json")
    assert got["ttft_ms"] == {"p50": 10.0, "p99": 30.0} and got["output_tokens_per_s"] == 900.0
    # SGLang appends one line per run: ours is the last one.
    lines = [json.dumps({**FLAT_RESULT, "median_ttft_ms": 1}), json.dumps(FLAT_RESULT)]
    (tmp_path / "s.jsonl").write_text("\n".join(lines) + "\n")
    assert ENGINE_BENCHMARKS["sglang"].read_result(tmp_path / "s.jsonl")["ttft_ms"]["p50"] == 10


class FakeBenchmark(EngineBenchmark):
    """Stands in for an engine's benchmark tool: writes a fixed result file."""

    name = "fake"
    engine = "fake"

    def write_dataset(self, prompts, path):
        path.write_text("\n".join(prompts))

    def command(self, run):
        code = f"open({str(run.result)!r}, 'w').write({json.dumps(FLAT_RESULT)!r})"
        return [sys.executable, "-c", code]

    def read_result(self, path):
        return ENGINE_BENCHMARKS["vllm"].read_result(path)


def test_validate_end_to_end(tmp_path, monkeypatch):
    requests = [Request(str(i), [{"role": "user", "content": f"p{i}"}], 4) for i in range(5)]
    monkeypatch.setitem(validation.ENGINE_BENCHMARKS, "fake", FakeBenchmark())
    monkeypatch.setattr(validation, "validation_requests", lambda *_: requests)
    cfg = validation.ValidationConfig(
        model="m",
        benchmarks=("fake",),
        num_requests=5,
        output_len=4,
        concurrency=(2,),
        runs=2,
        out=tmp_path / "p06.json",
    )
    report = asyncio.run(validation.validate_harness(cfg))
    rows = report["benchmarks"]["fake"]["levels"]["c2"]
    assert rows["ttft_ms.p50"]["theirs"] == 10.0 and rows["ttft_ms.p50"]["ours"] > 0
    assert rows["output_tokens_per_s"]["rel_diff"] is not None
    assert json.loads(cfg.out.read_text())["benchmarks"]["fake"]["levels"].keys() == {"c2"}
    # Every measured run of ours leaves per-request records next to the report.
    records = tmp_path / "p06" / "fake.c2.run0.ours.jsonl"
    lines = [json.loads(ln) for ln in records.read_text().splitlines()]
    assert len(lines) == len(requests) and {"ttft", "itl", "latency"} <= lines[0].keys()


def test_validation_requests_are_single_turn_with_fixed_output(monkeypatch):
    multi = Request("m", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], 9)
    single = [Request(str(i), [{"role": "user", "content": "u"}], 9) for i in range(3)]
    monkeypatch.setattr(validation, "load_workload", lambda w: (w, [multi, *single]))
    got = validation.validation_requests("W", 2, 256)
    assert [r.id for r in got] == ["0", "1"] and {r.max_tokens for r in got} == {256}
    with pytest.raises(HarnessError, match="only 3"):
        validation.validation_requests("W", 10, 256)


class FailingBenchmark(FakeBenchmark):
    name = "failing"

    def command(self, run):
        return [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]


def test_a_failing_benchmark_is_recorded_and_the_next_one_still_runs(tmp_path, monkeypatch):
    requests = [Request(str(i), [{"role": "user", "content": f"p{i}"}], 4) for i in range(3)]
    monkeypatch.setitem(validation.ENGINE_BENCHMARKS, "failing", FailingBenchmark())
    monkeypatch.setitem(validation.ENGINE_BENCHMARKS, "fake", FakeBenchmark())
    monkeypatch.setattr(validation, "validation_requests", lambda *_: requests)
    cfg = validation.ValidationConfig(
        model="m", benchmarks=("failing", "fake"), num_requests=3, output_len=4,
        concurrency=(1,), runs=1, out=tmp_path / "p06.json",
    )  # fmt: skip
    with pytest.raises(HarnessError, match="failing"):
        asyncio.run(validation.validate_harness(cfg))
    report = json.loads(cfg.out.read_text())
    assert "exited with code 3" in report["benchmarks"]["failing"]["error"]
    assert report["benchmarks"]["fake"]["levels"].keys() == {"c1"}

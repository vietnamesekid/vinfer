"""The ``run`` command end to end, against the fake server."""

from __future__ import annotations

import json

import pytest

from bench.harness import engines, result_file
from bench.harness import workload as workload_module
from bench.harness.__main__ import main
from bench.harness.fake_server import FakeServer


def test_run_writes_one_schema_valid_result_per_level(tmp_path):
    workload = tmp_path / "tiny.jsonl"
    rows = [
        {"id": str(i), "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        for i in range(6)
    ]
    workload.write_text("\n".join(json.dumps(r) for r in rows))
    paths = main(
        [
            "run",
            "--engine",
            "fake",
            "--workload-file",
            str(workload),
            "--concurrency",
            "1",
            "2",
            "--runs",
            "3",
            "--warmup",
            "1",
            "--allow-dirty",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert len(paths) == 2
    result = json.loads(paths[0].read_text())
    assert result["runs"] == 3 and len(result["per_run"]) == 3
    assert result["command"].endswith("--out-dir " + str(tmp_path / "out"))
    # Unofficial (fake engine, dirty tree, not dedicated), but never a schema gap.
    assert not [p for p in result_file.validate(result) if p.startswith("missing")]
    assert "unofficial" in str(paths[0])


def _run_fake_benchmark(tmp_path, monkeypatch, runs):
    """Run the ``run`` command on the fake server; return (that server, result files)."""
    servers = []

    class RecordingServer(FakeServer):
        def __init__(self):
            super().__init__(prefill_s=0.001, decode_s=0.001)
            servers.append(self)

    monkeypatch.setattr(engines, "FakeServer", RecordingServer)
    workload = tmp_path / "w.jsonl"
    rows = [
        {"id": str(i), "messages": [{"role": "user", "content": f"measured {i}"}], "max_tokens": 3}
        for i in range(6)
    ]
    workload.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = ["run", "--engine", "fake", "--workload-file", str(workload), "--concurrency", "2",
            "--runs", str(runs), "--warmup", "2", "--allow-dirty",
            "--out-dir", str(tmp_path / "out")]  # fmt: skip
    paths = main(args)
    return servers[0], paths


def test_prefix_cache_is_reset_after_warmup_and_before_every_run(tmp_path, monkeypatch):
    server, _ = _run_fake_benchmark(tmp_path, monkeypatch, runs=3)
    warmup = 2  # max(--warmup 2, concurrency 2)
    assert server.events == ["request"] * warmup + (["reset"] + ["request"] * 6) * 3
    # Warmup uses the workload's own prompt shapes; the reset right after it keeps that safe.
    assert all(p.startswith("measured") for p in server.prompts[:warmup])


def test_results_are_never_overwritten(tmp_path, monkeypatch):
    _, first = _run_fake_benchmark(tmp_path, monkeypatch, runs=1)
    _, second = _run_fake_benchmark(tmp_path, monkeypatch, runs=1)
    assert first[0] != second[0] and first[0].exists() and second[0].exists()
    result = json.loads(first[0].read_text())
    with pytest.raises(FileExistsError):
        result_file.write(result, tmp_path / "out", first[0].name.split("_")[0])


def test_one_server_launch_serves_every_workload(tmp_path, monkeypatch):
    for name in ("WA-v1", "WB-v1"):
        d = tmp_path / "workloads" / name
        d.mkdir(parents=True)
        row = {"id": "0", "messages": [{"role": "user", "content": name}], "max_tokens": 3}
        (d / "requests.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(workload_module, "WORKLOADS_DIR", tmp_path / "workloads")
    starts = []

    class CountingServer(FakeServer):
        def __init__(self):
            super().__init__(prefill_s=0.001, decode_s=0.001)
            starts.append(self)

    monkeypatch.setattr(engines, "FakeServer", CountingServer)
    paths = main(["run", "--engine", "fake", "--workload", "WA-v1", "WB-v1", "--concurrency",
                  "1", "--runs", "1", "--warmup", "1", "--allow-dirty",
                  "--out-dir", str(tmp_path / "out")])  # fmt: skip
    assert len(starts) == 1
    assert sorted(json.loads(p.read_text())["workload"] for p in paths) == ["WA-v1", "WB-v1"]

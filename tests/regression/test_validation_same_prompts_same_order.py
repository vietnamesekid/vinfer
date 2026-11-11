"""``validate`` must send our harness and each tool the same prompts in the same order.

Bug: SGLang's benchmark dropped prompts under 2 tokens (196 of 200 sent) and both tools
shuffled the prompts (seeded), while our harness sent all of them in file order. At a
concurrency above 1 the order decides which prompts are prefilled together, so TTFT was
compared on different batches.
"""

from __future__ import annotations

import asyncio
import random

from bench.harness import validation
from bench.harness.engine_benchmarks import ENGINE_BENCHMARKS, SEED
from bench.harness.workload import Request
from tests.unit.bench.test_validation import FakeBenchmark


def test_vllm_tool_keeps_the_dataset_order():
    assert ENGINE_BENCHMARKS["vllm"].send_order(5) == [0, 1, 2, 3, 4]
    assert "--disable-shuffle" in ENGINE_BENCHMARKS["vllm"].command(_run())


def test_sglang_order_is_its_seeded_shuffle():
    # What sglang.benchmark.serving does: random.seed(args.seed); random.shuffle(dataset).
    dataset = list(range(200))
    random.seed(SEED)
    random.shuffle(dataset)
    assert ENGINE_BENCHMARKS["sglang"].send_order(200) == dataset
    assert "--seed" in (command := ENGINE_BENCHMARKS["sglang"].command(_run()))
    assert command[command.index("--seed") + 1] == str(SEED)


def test_short_prompts_are_dropped_for_every_side(monkeypatch):
    requests = [Request(str(i), [{"role": "user", "content": p}], 9) for i, p in enumerate(
        ["hi", "a longer prompt", "x", "another longer one"]
    )]  # fmt: skip
    monkeypatch.setattr(validation, "load_workload", lambda w: (w, requests))
    monkeypatch.setattr(validation, "_token_counter", lambda *_: lambda text: len(text.split()))
    got = validation.validation_requests("W", 2, 16, tokenizer="m")
    assert [r.id for r in got] == ["1", "3"]


class ReversedBenchmark(FakeBenchmark):
    name = "reversed"

    def send_order(self, n):
        return list(reversed(range(n)))


def test_ours_sends_in_the_tool_order(tmp_path, monkeypatch):
    requests = [Request(str(i), [{"role": "user", "content": f"p{i}"}], 4) for i in range(4)]
    monkeypatch.setitem(validation.ENGINE_BENCHMARKS, "reversed", ReversedBenchmark())
    monkeypatch.setattr(validation, "validation_requests", lambda *_: requests)
    sent = []
    real_send = validation.send_requests

    async def recording_send(url, model, reqs, **kw):
        sent.append([r.id for r in reqs])
        return await real_send(url, model, reqs, **kw)

    monkeypatch.setattr(validation, "send_requests", recording_send)
    cfg = validation.ValidationConfig(
        model="m", benchmarks=("reversed",), num_requests=4, output_len=4,
        concurrency=(2,), runs=1, out=tmp_path / "p06.json",
    )  # fmt: skip
    asyncio.run(validation.validate_harness(cfg))
    assert sent[-1] == ["3", "2", "1", "0"]  # the measured run, after the warmup


def _run():
    from tests.unit.bench.test_validation import RUN

    return RUN

"""The OpenAI-compatible server on a tiny model (CPU), measured by the bench harness's own client.

The server runs in uvicorn on a free local port, exactly as the harness will see it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest

from bench.harness.http_client import stream_chat_completion
from bench.harness.workload import Request
from tests.unit.models import fixtures
from vinfer.runtime.tokenizer import Tokenizer
from vinfer.scheduler.serial import SerialScheduler
from vinfer.server.app import create_app

aiohttp = pytest.importorskip("aiohttp")
uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("transformers")

MESSAGES = [{"role": "user", "content": "w1 w2 w3 w40"}]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    path = fixtures.save(fixtures.hf_qwen3_5(), tmp_path_factory.mktemp("srv"))
    fixtures.save_tiny_tokenizer(path)
    model = fixtures.load(path)
    tokenizer = Tokenizer(path)
    pool = model.new_state_pool(num_slots=1, max_seq_len=128)
    scheduler = SerialScheduler(model, pool, tokenizer.eos_token_ids, tokenizer.decode)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(scheduler, tokenizer, "tiny"), port=port))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started
    yield {"url": f"http://127.0.0.1:{port}", "pool": pool, "tokenizer": tokenizer}
    srv.should_exit = True
    thread.join(timeout=10)
    scheduler.close()


def _post(url: str, body: dict):
    async def go():
        async with aiohttp.ClientSession() as s, s.post(url, json=body) as r:
            return r.status, await r.read()

    return asyncio.run(go())


def test_harness_measures_a_streamed_request(server):
    async def go():
        async with aiohttp.ClientSession() as session:
            return await stream_chat_completion(
                session,
                server["url"] + "/v1/chat/completions",
                "tiny",
                Request("r0", MESSAGES, 12),
            )

    timing = asyncio.run(go())
    assert timing.success, timing.error
    assert timing.output_tokens == 12 and len(timing.itl) == 11  # one chunk per token
    assert timing.ttft > 0 and timing.prompt_tokens == len(server["tokenizer"].chat_ids(MESSAGES))
    assert server["pool"].num_free == 1


def test_non_streaming_response(server):
    body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 5, "temperature": 0}
    status, raw = _post(server["url"] + "/v1/chat/completions", body)
    out = json.loads(raw)
    assert status == 200 and out["usage"]["completion_tokens"] <= 5
    assert out["choices"][0]["finish_reason"] in {"stop", "length"}


def test_greedy_is_deterministic_and_seeded_sampling_reproducible(server):
    url = server["url"] + "/v1/chat/completions"
    for extra in ({"temperature": 0}, {"temperature": 1.0, "seed": 11}):
        body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 8, "ignore_eos": True, **extra}
        a, b = (json.loads(_post(url, body)[1])["choices"][0]["message"] for _ in range(2))
        assert a == b


def test_bad_requests_get_400(server):
    url = server["url"] + "/v1/chat/completions"
    assert _post(url, {"model": "tiny"})[0] == 400  # no messages
    assert _post(url, {"model": "tiny", "messages": MESSAGES, "n": 2})[0] == 400
    assert _post(url, {"model": "tiny", "messages": MESSAGES, "temperature": -1})[0] == 400


def test_client_disconnect_frees_the_slot(server):
    body = {
        "model": "tiny",
        "messages": MESSAGES,
        "max_tokens": 100,
        "ignore_eos": True,
        "stream": True,
    }

    async def go():
        async with (
            aiohttp.ClientSession() as s,
            s.post(server["url"] + "/v1/chat/completions", json=body) as r,
        ):
            n = 0
            async for line in r.content:
                n += line.startswith(b"data:")
                if n == 3:
                    break  # leave after a couple of tokens

    asyncio.run(go())
    deadline = time.time() + 10
    while server["pool"].num_free != 1 and time.time() < deadline:
        time.sleep(0.05)
    assert server["pool"].num_free == 1

"""Workload generator: determinism and structure, with a whitespace tokenizer. CPU only."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json

import pytest

from bench.workloads.generate import SPECS, build, to_jsonl


class WsTok:
    def count(self, text: str) -> int:
        return len(text.split())

    def count_chat(self, messages) -> int:
        return sum(self.count(m["content"]) + 4 for m in messages) + 3

    def truncate(self, text: str, n: int) -> str:
        return " ".join(text.split()[:n])


def _sharegpt(n: int, turns: int = 4):
    return [
        {
            "id": f"c{i}",
            "conversations": [
                x
                for t in range(turns)
                for x in (
                    {"from": "human", "value": f"question {i} {t} " + "w " * (5 + i % 30)},
                    {"from": "gpt", "value": "answer " * (10 + (i * 7 + t) % 400)},
                )
            ],
        }
        for i in range(n)
    ]


def _spec(wid: str, **kw):
    return dataclasses.replace(SPECS[wid], **kw)


def test_chat_is_deterministic_and_respects_lengths():
    spec = _spec("W1-v1", num_requests=50)
    rows = _sharegpt(200)
    a, b = to_jsonl(build(spec, rows, WsTok())), to_jsonl(build(spec, rows, WsTok()))
    assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
    reqs = [json.loads(x) for x in a.splitlines()]
    assert len(reqs) == 50 and len({r["source"] for r in reqs}) == 50
    lo, hi = spec.output_tokens
    assert all(lo <= r["max_tokens"] <= hi for r in reqs)
    c = to_jsonl(build(dataclasses.replace(spec, seed=1), rows, WsTok()))
    assert c != a


def test_multi_turn_sessions_share_prefix_and_grow():
    spec = _spec("W4-v1", num_requests=10)
    reqs = build(spec, _sharegpt(100, turns=6), WsTok())
    sessions = {}
    for r in reqs:
        sessions.setdefault(r["session_id"], []).append(r)
    assert len(sessions) == 10
    system = reqs[0]["messages"][0]
    for turns in sessions.values():
        assert [t["turn"] for t in turns] == list(range(len(turns)))
        assert len(turns) >= spec.turns[0]
        for prev, cur in itertools.pairwise(turns):
            # Turn k+1 extends turn k's prompt with the recorded reply and a new user turn.
            assert cur["messages"][: len(prev["messages"])] == prev["messages"]
            assert cur["messages"][len(prev["messages"])]["role"] == "assistant"
        assert all(t["messages"][0] == system for t in turns)


def test_long_doc_truncates_to_target():
    rows = [
        {
            "_id": str(i),
            "context": "tok " * 40000,
            "question": "q?",
            "choice_A": "a",
            "choice_B": "b",
            "choice_C": "c",
            "choice_D": "d",
        }
        for i in range(30)
    ]
    reqs = build(_spec("W2-v1", num_requests=20), rows, WsTok())
    assert {r["doc_tokens"] for r in reqs} <= set(SPECS["W2-v1"].doc_tokens)
    for r in reqs:
        assert r["prompt_tokens"] >= r["doc_tokens"]


def test_not_enough_rows_fails_loudly():
    with pytest.raises(ValueError, match="only"):
        build(_spec("W1-v1", num_requests=500), _sharegpt(10), WsTok())

"""Generate the fixed, seeded workloads W1-W4.

Usage::

    python -m bench.workloads.generate W1-v1 [W2-v1 ...]      # write requests.jsonl + manifest
    python -m bench.workloads.generate --check W1-v1           # regenerate, compare sha256

Output per workload: ``bench/workloads/<id>/requests.jsonl`` (git-ignored: large, and some
sources should not be redistributed) and ``manifest.json`` (committed: spec, source revisions,
tokenizer revision, sha256 of the jsonl). Same spec + same sources = byte-identical jsonl.

Never edit a version in place. Changing anything in a spec means a new version (``W1-v2``).
Token counts use the Qwen3.5 tokenizer with its chat template, as the server will see them.

Request fields: ``id``, ``messages``, ``max_tokens`` (output length, forced by the harness),
``prompt_tokens``, ``session_id``/``turn`` (W4 only), ``source`` (dataset row it came from).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

HERE = Path(__file__).resolve().parent

TOKENIZER = {"repo": "Qwen/Qwen3.5-2B", "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"}
SOURCES = {
    "sharegpt": {
        "repo": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "revision": "192ab2185289094fc556ec8ce5ce1e8e587154ca",
        "file": "ShareGPT_V3_unfiltered_cleaned_split.json",
        "license": "apache-2.0 (HF tag; content provenance unclear, not redistributed)",
    },
    "longbench_v2": {
        "repo": "zai-org/LongBench-v2",
        "revision": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        "file": "data.json",
        "license": "apache-2.0",
    },
    "humaneval": {
        "repo": "openai/openai_humaneval",
        "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "file": "openai_humaneval/test-00000-of-00001.parquet",
        "license": "mit",
    },
    "mbpp": {
        "repo": "google-research-datasets/mbpp",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "file": "sanitized/test-00000-of-00001.parquet",
        "license": "cc-by-4.0",
    },
}


class Tokenizer(Protocol):
    def count_chat(self, messages: list[dict[str, str]]) -> int: ...
    def count(self, text: str) -> int: ...
    def truncate(self, text: str, n_tokens: int) -> str: ...


@dataclass(frozen=True)
class Spec:
    """Everything that defines a workload version. Lengths are in tokens."""

    id: str
    kind: str  # "chat" | "long_doc" | "code" | "multi_turn"
    sources: tuple[str, ...]
    num_requests: int  # W4: number of sessions
    seed: int = 0
    prompt_tokens: tuple[int, int] = (4, 1024)  # inclusive filter on the prompt
    output_tokens: tuple[int, int] = (4, 512)  # clip (chat, multi_turn) or uniform (others)
    doc_tokens: tuple[int, ...] = ()  # long_doc: context lengths to sample from
    turns: tuple[int, int] = (3, 6)  # multi_turn: turns per session (min, max)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


SPECS: dict[str, Spec] = {
    s.id: s
    for s in [
        Spec(
            id="W1-v1",
            kind="chat",
            sources=("sharegpt",),
            num_requests=1000,
            prompt_tokens=(4, 1024),
            output_tokens=(4, 512),
            notes="First user turn of ShareGPT conversations; output length = length of the "
            "recorded first reply, clipped. Like vLLM's sharegpt sampling, shorter.",
        ),
        Spec(
            id="W2-v1",
            kind="long_doc",
            sources=("longbench_v2",),
            num_requests=200,
            doc_tokens=(4096, 8192, 16384, 32768),
            output_tokens=(64, 256),
            notes="LongBench-v2 context truncated (head kept) to a length drawn uniformly from "
            "doc_tokens, followed by its question; output length uniform in output_tokens.",
        ),
        Spec(
            id="W3-v1",
            kind="code",
            sources=("humaneval", "mbpp"),
            num_requests=400,
            prompt_tokens=(4, 1024),
            output_tokens=(1024, 2048),
            notes="HumanEval and MBPP (sanitized test) task prompts as code requests; output "
            "length uniform in output_tokens.",
        ),
        Spec(
            id="W4-v1",
            kind="multi_turn",
            sources=("sharegpt",),
            num_requests=200,
            turns=(3, 6),
            prompt_tokens=(4, 16384),
            output_tokens=(16, 512),
            notes="Agent loop: one shared system prompt (rules + 24 JSON tool schemas, "
            "~3.3k tokens) + ShareGPT conversations with >= 3 user turns. Turn k sends the "
            "system prompt and the recorded history up to turn k; output length = recorded "
            "reply k, clipped.",
        ),
    ]
}


# ----------------------------------------------------------------------------- builders


def _clip(n: int, lo_hi: tuple[int, int]) -> int:
    return max(lo_hi[0], min(lo_hi[1], n))


def _sharegpt_turns(row: dict[str, Any]) -> list[tuple[str, str]]:
    """(user, assistant) pairs from a ShareGPT row, stopping at the first malformed pair."""
    conv = row.get("conversations") or []
    pairs = []
    for i in range(0, len(conv) - 1, 2):
        u, a = conv[i], conv[i + 1]
        if u.get("from") != "human" or a.get("from") != "gpt":
            break
        if not u.get("value", "").strip() or not a.get("value", "").strip():
            break
        pairs.append((u["value"], a["value"]))
    return pairs


def build_chat(spec: Spec, rows: Sequence[dict], tok: Tokenizer, rng: np.random.Generator):
    out = []
    for idx in rng.permutation(len(rows)):
        if len(out) == spec.num_requests:
            break
        pairs = _sharegpt_turns(rows[idx])
        if not pairs:
            continue
        user, reply = pairs[0]
        messages = [{"role": "user", "content": user}]
        n_prompt = tok.count_chat(messages)
        n_out = tok.count(reply)
        if not spec.prompt_tokens[0] <= n_prompt <= spec.prompt_tokens[1] or n_out < 4:
            continue
        out.append(
            {
                "messages": messages,
                "max_tokens": _clip(n_out, spec.output_tokens),
                "prompt_tokens": n_prompt,
                "source": f"sharegpt:{rows[idx].get('id', idx)}",
            }
        )
    return out


def build_long_doc(spec: Spec, rows: Sequence[dict], tok: Tokenizer, rng: np.random.Generator):
    out = []
    order = rng.permutation(len(rows))
    i = 0
    while len(out) < spec.num_requests and i < len(order):
        row = rows[order[i]]
        i += 1
        target = int(rng.choice(spec.doc_tokens))
        if tok.count(row["context"]) < target:
            continue
        context = tok.truncate(row["context"], target)
        question = (
            f"{row['question']}\n\nA. {row['choice_A']}\nB. {row['choice_B']}\n"
            f"C. {row['choice_C']}\nD. {row['choice_D']}"
        )
        content = f"Read the document and answer the question.\n\n<document>\n{context}\n"
        content += f"</document>\n\nQuestion: {question}"
        messages = [{"role": "user", "content": content}]
        out.append(
            {
                "messages": messages,
                "max_tokens": int(rng.integers(spec.output_tokens[0], spec.output_tokens[1] + 1)),
                "prompt_tokens": tok.count_chat(messages),
                "doc_tokens": target,
                "source": f"longbench_v2:{row.get('_id', order[i - 1])}",
            }
        )
    return out


def build_code(spec: Spec, rows: Sequence[dict], tok: Tokenizer, rng: np.random.Generator):
    out = []
    for idx in rng.permutation(len(rows)):
        if len(out) == spec.num_requests:
            break
        row = rows[idx]
        if row["_source"] == "humaneval":
            content = (
                "Complete the following Python function. Explain your approach, then give the "
                f"full implementation and tests.\n\n```python\n{row['prompt']}```"
            )
        else:
            content = (
                f"{row['prompt']}\n\nWrite a Python solution. Explain your approach, then give "
                "the full implementation and tests."
            )
        messages = [{"role": "user", "content": content}]
        n_prompt = tok.count_chat(messages)
        if not spec.prompt_tokens[0] <= n_prompt <= spec.prompt_tokens[1]:
            continue
        out.append(
            {
                "messages": messages,
                "max_tokens": int(rng.integers(spec.output_tokens[0], spec.output_tokens[1] + 1)),
                "prompt_tokens": n_prompt,
                "source": f"{row['_source']}:{row['task_id']}",
            }
        )
    return out


def agent_system_prompt() -> str:
    """Deterministic agent system prompt with JSON tool schemas (shared prefix for W4)."""
    tools = []
    verbs = ["search", "read", "write", "list", "delete", "update", "run", "fetch", "summarize"]
    nouns = ["file", "issue", "ticket", "page", "record", "message", "calendar_event", "metric"]
    for v in verbs:
        for n in nouns[: 2 + len(v) % 3]:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"{v}_{n}",
                        "description": f"{v.capitalize()} a {n.replace('_', ' ')} in the user's "
                        "workspace. Returns a JSON object with `ok`, `data` and `error` fields. "
                        "Call it only when the user's request needs it.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": f"The {n} identifier."},
                                "query": {"type": "string", "description": "Free-text filter."},
                                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                            },
                            "required": ["id"] if v != "search" else ["query"],
                        },
                    },
                }
            )
    rules = (
        "You are an assistant that helps the user with their workspace. Think about what the "
        "user needs, call tools when they help, and answer clearly. Never invent tool results. "
        "If a tool fails, say so and suggest a next step. Keep answers concise unless the user "
        "asks for detail."
    )
    return rules + "\n\n# Tools\n\n" + "\n".join(json.dumps(t, sort_keys=True) for t in tools)


def build_multi_turn(spec: Spec, rows: Sequence[dict], tok: Tokenizer, rng: np.random.Generator):
    system = {"role": "system", "content": agent_system_prompt()}
    out = []
    sessions = 0
    for idx in rng.permutation(len(rows)):
        if sessions == spec.num_requests:
            break
        pairs = _sharegpt_turns(rows[idx])
        if len(pairs) < spec.turns[0]:
            continue
        n_turns = min(len(pairs), int(rng.integers(spec.turns[0], spec.turns[1] + 1)))
        history: list[dict[str, str]] = [system]
        reqs = []
        for t in range(n_turns):
            user, reply = pairs[t]
            messages = [*history, {"role": "user", "content": user}]
            n_prompt = tok.count_chat(messages)
            if n_prompt > spec.prompt_tokens[1]:
                break
            reqs.append(
                {
                    "messages": messages,
                    "max_tokens": _clip(tok.count(reply), spec.output_tokens),
                    "prompt_tokens": n_prompt,
                    "session_id": f"s{sessions}",
                    "turn": t,
                    "source": f"sharegpt:{rows[idx].get('id', idx)}#{t}",
                }
            )
            history = [*messages, {"role": "assistant", "content": reply}]
        if len(reqs) < spec.turns[0]:
            continue
        out.extend(reqs)
        sessions += 1
    return out


BUILDERS: dict[str, Callable] = {
    "chat": build_chat,
    "long_doc": build_long_doc,
    "code": build_code,
    "multi_turn": build_multi_turn,
}


def build(spec: Spec, rows: Sequence[dict], tok: Tokenizer) -> list[dict[str, Any]]:
    rng = np.random.default_rng(spec.seed)
    reqs = BUILDERS[spec.kind](spec, rows, tok, rng)
    if spec.kind != "multi_turn" and len(reqs) < spec.num_requests:
        raise ValueError(f"{spec.id}: only {len(reqs)} of {spec.num_requests} requests")
    for i, r in enumerate(reqs):
        r["id"] = f"{spec.id}-{i:05d}"
    return reqs


def to_jsonl(reqs: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True).encode() + b"\n" for r in reqs
    )


# ----------------------------------------------------------------------------- real I/O


class HFTokenizer:
    def __init__(self) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(TOKENIZER["repo"], revision=TOKENIZER["revision"])

    def count_chat(self, messages: list[dict[str, str]]) -> int:
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return len(ids)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def truncate(self, text: str, n_tokens: int) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tok.decode(ids)


def load_rows(spec: Spec) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, Any]] = []
    for name in spec.sources:
        src = SOURCES[name]
        path = hf_hub_download(
            src["repo"], src["file"], repo_type="dataset", revision=src["revision"]
        )
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            part = pq.read_table(path).to_pylist()
        else:
            part = json.loads(Path(path).read_text())
        for r in part:
            r["_source"] = name
        rows.extend(part)
    return rows


def generate(wid: str, check: bool = False) -> str:
    """Build workload ``wid`` and write its ``requests.jsonl``.

    - Existing version (``manifest.json`` committed): the rebuilt file must have the sha256 in
      the manifest, else nothing is written. The manifest is never rewritten.
    - New version (no manifest yet): write the jsonl and create the manifest.
    - ``check=True``: only compare with the manifest, write nothing.
    """
    spec = SPECS[wid]
    reqs = build(spec, load_rows(spec), HFTokenizer())
    data = to_jsonl(reqs)
    sha = hashlib.sha256(data).hexdigest()
    d = HERE / wid
    manifest_path = d / "manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())["sha256"]
        if sha != expected:
            sys.exit(f"{wid}: rebuilt sha256 {sha} != manifest {expected}; nothing written")
        if not check:
            (d / "requests.jsonl").write_bytes(data)
        print(f"{wid}: OK {sha}" + ("" if check else " (requests.jsonl written)"))
        return sha
    if check:
        sys.exit(f"{wid}: no manifest to check against")
    d.mkdir(parents=True, exist_ok=True)
    (d / "requests.jsonl").write_bytes(data)
    manifest_path.write_text(json.dumps(_manifest(wid, spec, reqs, sha), indent=2) + "\n")
    print(f"{wid}: new version, {len(reqs)} requests, sha256 {sha}")
    return sha


def _manifest(wid: str, spec: Spec, reqs: list[dict[str, Any]], sha: str) -> dict[str, Any]:
    sessions = {r.get("session_id") for r in reqs} if spec.kind == "multi_turn" else None
    return {
        "id": wid,
        "spec": asdict(spec),
        "sources": {n: SOURCES[n] for n in spec.sources},
        "tokenizer": TOKENIZER,
        "num_requests": len(reqs),
        "num_sessions": len(sessions) if sessions is not None else None,
        "prompt_tokens": _summary(np.array([r["prompt_tokens"] for r in reqs])),
        "output_tokens": _summary(np.array([r["max_tokens"] for r in reqs])),
        "sha256": sha,
    }


def _summary(x: np.ndarray) -> dict[str, float]:
    return {
        "min": int(x.min()),
        "p50": float(np.percentile(x, 50)),
        "mean": float(x.mean()),
        "p99": float(np.percentile(x, 99)),
        "max": int(x.max()),
        "sum": int(x.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("workloads", nargs="*", default=list(SPECS))
    p.add_argument("--check", action="store_true", help="regenerate and compare with manifest")
    a = p.parse_args()
    for w in a.workloads:
        generate(w, check=a.check)


if __name__ == "__main__":
    main()

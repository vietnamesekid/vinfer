"""Load and tokenize the fixed parity prompt set (``prompts/v1.jsonl``).

Kinds: ``short``, ``multilingual``, ``code``, ``chat`` (system + multi-turn), ``math`` go through
the model's chat template; ``raw`` is tokenized as plain text; ``long`` concatenates snapshot
files in ``prompts/long/`` (frozen copies, so the set never changes) and appends an instruction.
Never edit v1: add v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent / "prompts"


def load(version: str = "v1") -> list[dict[str, Any]]:
    return [json.loads(x) for x in (HERE / f"{version}.jsonl").read_text().splitlines() if x]


def tokenize(prompts: list[dict[str, Any]], tokenizer: Any) -> dict[str, list[int]]:
    out = {}
    for p in prompts:
        if p["kind"] == "raw":
            ids = tokenizer.encode(p["text"], add_special_tokens=False)
        else:
            if p["kind"] == "long":
                doc = "\n\n".join((HERE / "long" / f).read_text() for f in p["files"])
                messages = [{"role": "user", "content": f"{doc}\n\n{p['instruction']}"}]
            else:
                messages = p["messages"]
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
        out[p["id"]] = list(ids)
    return out

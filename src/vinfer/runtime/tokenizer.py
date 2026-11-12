"""Tokenizer and chat template: HF Transformers' tokenizer only, never its models.

Using the same ``AutoTokenizer`` and ``apply_chat_template`` as the parity tools and the bench
harness means a prompt becomes exactly the same token ids everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Tokenizer:
    def __init__(self, path: str | Path) -> None:
        from transformers import AutoTokenizer

        self.path = Path(path)
        self._tok = AutoTokenizer.from_pretrained(str(path))
        self.eos_token_ids = frozenset(self._eos_ids())

    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        if self._tok.eos_token_id is not None:
            ids.add(self._tok.eos_token_id)
        for name in ("generation_config.json", "config.json"):
            file = self.path / name
            if not file.exists():
                continue
            raw = json.loads(file.read_text())
            for cfg in (raw, raw.get("text_config", {})):
                value = cfg.get("eos_token_id")
                ids.update(value if isinstance(value, list) else [] if value is None else [value])
        return ids

    def chat_ids(self, messages: list[dict[str, Any]], **template_kwargs: Any) -> list[int]:
        """Token ids of ``messages`` through the chat template. A template that rejects the
        messages (Qwen's call ``raise_exception``) raises ``ValueError``: it is the request's
        fault, not the server's."""
        from jinja2 import TemplateError

        try:
            ids = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **template_kwargs
            )
        except TemplateError as exc:
            raise ValueError(f"chat template rejected the messages: {exc}") from exc
        if hasattr(ids, "keys"):  # some versions return a BatchEncoding
            ids = ids["input_ids"]
        return list(ids)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

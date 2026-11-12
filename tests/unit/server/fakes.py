"""A scheduler that "generates" given text pieces, and a tokenizer that records what the server
asks of it, to test exactly how the server presents a known output."""

from __future__ import annotations

from typing import Any

from vinfer.runtime import SamplingParams, Step


class FakeTokenizer:
    """``chat_ids`` returns ``prompt_len`` ids and remembers its arguments; the prompt decodes
    to ``prompt_text`` (with special tokens) whatever the ids."""

    def __init__(self, prompt_text: str = "<|im_start|>assistant\n", prompt_len: int = 5) -> None:
        self.prompt_text = prompt_text
        self.prompt_len = prompt_len
        self.calls: list[dict[str, Any]] = []

    def chat_ids(self, messages, tools=None, **kwargs) -> list[int]:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if kwargs.get("fail"):
            raise ValueError("chat template rejected the messages: fail")
        return list(range(self.prompt_len))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.prompt_text if not skip_special_tokens else ""


class FakeScheduler:
    """Yields one ``Step`` per piece of ``pieces``, the last with ``finish_reason``. A request's
    ``max_tokens`` cuts the output short with ``"length"``, as the runtime does."""

    def __init__(self, pieces: list[str] | None = None, finish: str = "stop") -> None:
        self.pieces = pieces or ["Hello", " world"]
        self.finish = finish
        self.max_seq_len = 1024
        self.params: SamplingParams | None = None
        self.fail_after: int | None = None  # raise after this many steps

    async def stream(self, prompt_ids: list[int], params: SamplingParams):
        self.params = params
        pieces = self.pieces[: params.max_tokens]
        cut = len(pieces) < len(self.pieces)
        for i, text in enumerate(pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("boom")
            last = i == len(pieces) - 1
            yield Step(i, text, ("length" if cut else self.finish) if last else None)

"""Prefill/decode loop for one sequence (batch size 1), with stop conditions and streaming text.

The runtime asks the state pool for a slot and gives it back; it never allocates cache memory
itself.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

from vinfer.models.base import CausalLM
from vinfer.runtime.sampler import SamplingParams, sample
from vinfer.state import StatePool


@dataclass
class Step:
    token_id: int
    text: str  # new text safe to show (may be empty while it could still be a stop string)
    finish_reason: str | None = None  # "stop" | "length" | None


class _TextStream:
    """Turns a growing token list into text deltas and finds stop strings.

    Holds back text that might still become a stop string, and an unfinished UTF-8 sequence
    (decoded as U+FFFD), so streamed text never has to be taken back.
    """

    def __init__(self, decode: Callable[[list[int]], str], stops: tuple[str, ...]) -> None:
        self.decode = decode
        self.stops = tuple(s for s in stops if s)
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.tokens: list[int] = []
        self.sent = 0  # characters already emitted

    def push(self, token_id: int | None) -> tuple[str, bool]:
        """Add a token (``None``: no token, flush). Returns ``(delta, hit_stop)``."""
        if token_id is not None:
            self.tokens.append(token_id)
        text = self.decode(self.tokens)
        start = max(0, self.sent - self.hold)
        hits = [i for i in (text.find(s, start) for s in self.stops) if i >= 0]
        if hits:
            cut = min(hits)
            delta = text[self.sent : cut]
            self.sent = max(self.sent, cut)
            return delta, True
        if token_id is None:  # final flush: everything left
            delta, self.sent = text[self.sent :], len(text)
            return delta, False
        safe = len(text) - self.hold
        if text.endswith("�"):
            safe = min(safe, len(text) - 1)
        if safe <= self.sent:
            return "", False
        delta, self.sent = text[self.sent : safe], safe
        return delta, False

    def flush(self) -> str:
        """Everything held back, up to a stop string if one is found."""
        return self.push(None)[0]


def generate(
    model: CausalLM,
    pool: StatePool,
    prompt_ids: list[int],
    params: SamplingParams,
    eos_token_ids: frozenset[int] = frozenset(),
    decode: Callable[[list[int]], str] | None = None,
) -> Iterator[Step]:
    """Yield one ``Step`` per generated token; the last one has a ``finish_reason``.

    ``decode`` turns token ids into text; without it, ``Step.text`` is empty and stop strings
    are not checked.
    """
    if not prompt_ids:
        raise ValueError("empty prompt")
    if len(prompt_ids) + params.max_tokens > pool.max_seq_len:
        raise ValueError(
            f"prompt ({len(prompt_ids)}) + max_tokens ({params.max_tokens}) exceeds "
            f"max_seq_len ({pool.max_seq_len})"
        )
    seed = params.seed if params.seed is not None else random.getrandbits(63)
    stream = _TextStream(decode, params.stop) if decode else None
    stop_ids = set(params.stop_token_ids) | (set() if params.ignore_eos else set(eos_token_ids))
    device = model.device
    slot = pool.alloc()
    try:
        ids = torch.tensor(prompt_ids, device=device)
        position = len(prompt_ids)
        logits = model.logits(model.forward_hidden(ids, 0, slot, pool)[-1:])[0]
        for n in range(1, params.max_tokens + 1):
            token = sample(logits, params, seed, position)
            if token in stop_ids:  # the stop token's own text is not part of the output
                yield Step(token, stream.flush() if stream else "", "stop")
                return
            text, hit = stream.push(token) if stream else ("", False)
            if hit:
                yield Step(token, text, "stop")
                return
            if n == params.max_tokens:
                yield Step(token, text + (stream.flush() if stream else ""), "length")
                return
            yield Step(token, text)
            next_ids = torch.tensor([token], device=device)
            hidden = model.forward_hidden(next_ids, position, slot, pool)
            logits = model.logits(hidden)[0]
            position += 1
    finally:
        pool.free(slot)

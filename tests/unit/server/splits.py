"""Feed a parser the same text in every possible way and check it always gives one result."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _runs(text: str):
    yield [text]
    yield list(text)  # one character at a time
    for i in range(len(text) + 1):
        yield [text[:i], text[i:]]
    for i in range(0, len(text) + 1, 3):  # three pieces, the cut moving through the text
        for j in range(i, len(text) + 1, 5):
            yield [text[:i], text[i:j], text[j:]]


def assert_same_for_every_split(
    new: Callable[[], Any], text: str, collect: Callable[[list[Any]], Any]
) -> Any:
    """``new()`` makes a parser; ``collect(deltas)`` reduces its ``feed``/``finish`` deltas to a
    comparable result. Returns the result of feeding ``text`` whole."""
    expected = None
    for pieces in _runs(text):
        parser = new()
        deltas = [parser.feed(p) for p in pieces] + [parser.finish()]
        got = collect(deltas)
        if expected is None:
            expected = got
        assert got == expected, f"pieces {pieces!r}: {got!r} != {expected!r}"
    return expected

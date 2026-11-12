"""Streaming text must not re-decode the whole output at every token.

Bug: each step decoded every generated token again (O(n^2) over the output), so the time per
token grew with the output length because of the tokenizer, not the model, skewing TPOT on
long-output workloads.
"""

from __future__ import annotations

from vinfer.runtime.generate import _TextStream


def test_each_step_decodes_a_short_window_and_the_text_is_complete():
    seen: list[int] = []

    def decode(ids: list[int]) -> str:
        seen.append(len(ids))
        return "".join(chr(97 + i % 26) for i in ids)

    stream = _TextStream(decode, ("zzz",))
    text = "".join(stream.push(i)[0] for i in range(500)) + stream.flush()
    assert text == "".join(chr(97 + i % 26) for i in range(500))
    assert max(seen) <= 4, max(seen)

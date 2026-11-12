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


def test_tokens_that_decode_to_nothing_do_not_grow_the_window():
    # Bug: the window only advanced when new text appeared, so a run of tokens decoding to ""
    # (skipped special tokens) made every step decode the whole run again.
    seen: list[int] = []

    def decode(ids: list[int]) -> str:
        seen.append(len(ids))
        return "".join("x" for i in ids if i == 1)  # token 0 decodes to nothing

    stream = _TextStream(decode, ())
    text = "".join(stream.push(0)[0] for _ in range(300)) + stream.push(1)[0] + stream.flush()
    assert text == "x"
    assert max(seen) <= 4, max(seen)

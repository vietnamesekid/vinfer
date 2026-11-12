"""Tool calls in Qwen3.5's XML format ("qwen3_coder"), from its chat template::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

Follows SGLang's ``Qwen3CoderDetector`` and vLLM's ``qwen3`` grammar: one leading and one
trailing newline of a value are markup; a value ends at ``</parameter>``, or at the next
``<parameter=`` or ``</function>`` if the model forgot to close it; a bare ``<function=``
without ``<tool_call>`` is accepted; several functions may share one ``<tool_call>``. Values
are typed by the tool's JSON schema.

Streaming sends the name as soon as ``<function=NAME>`` is complete, then each parameter as a
JSON fragment once it is closed (``{"k": v``, ``, "k2": v2``, ``}``), so the arguments of a
call concatenate to ``json.dumps`` of its parameters.
"""

from __future__ import annotations

import enum
import json
from typing import Any

from vinfer.server.parsers.base import ToolCallDelta, ToolCallParser, ToolDelta, find_tag
from vinfer.server.parsers.registry import TOOL_CALL_PARSERS
from vinfer.server.parsers.tool_text import ContentAround, coerce_value

TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"
FUNC_START, FUNC_END = "<function=", "</function>"
PARAM_START, PARAM_END = "<parameter=", "</parameter>"


class _State(enum.Enum):
    TEXT = 0  # content
    CALL = 1  # after <tool_call>, before <function=
    NAME = 2  # after <function=, before >
    ARGS = 3  # inside a function, between parameters
    PARAM = 4  # after <parameter=
    AFTER = 5  # after </function>, before </tool_call>


def _trim_markup_newlines(value: str) -> str:
    value = value.removeprefix("\n")
    return value.removesuffix("\n")


@TOOL_CALL_PARSERS.register("qwen3_coder")
class Qwen3CoderToolCallParser(ToolCallParser):
    start_tags = (TOOL_START, FUNC_START)

    def __init__(self, tools=()) -> None:
        super().__init__(tools)
        self._state = _State.TEXT
        self._buf = ""
        self._content = ContentAround()
        self._call: ToolCallDelta | None = None  # the open call
        self._params: dict[str, Any] = {}
        self._n_params = 0

    def feed(self, text: str) -> ToolDelta:
        self._buf += text
        out = ToolDelta()
        while self._step(out, final=False):
            pass
        return out

    def finish(self) -> ToolDelta:
        out = ToolDelta()
        while self._step(out, final=True):
            pass
        if self._state is _State.TEXT:
            out.content += self._content.text(self._buf)
        self._buf = ""
        self._close_call(out)
        out.content += self._content.finish()
        return out

    # -- calls --

    def _open_call(self, name: str, out: ToolDelta) -> None:
        self._content.call()
        self._call = self._new_call(name.strip())
        self._params, self._n_params = self._call_schema(), 0
        out.calls.append(self._call)

    def _call_schema(self) -> dict[str, Any]:
        assert self._call is not None and self._call.name is not None
        return self.parameter_schemas(self._call.name)

    def _add_param(self, name: str, raw: str, out: ToolDelta) -> None:
        assert self._call is not None
        value = coerce_value(_trim_markup_newlines(raw), self._params.get(name))
        piece = f"{json.dumps(name.strip(), ensure_ascii=False)}: "
        piece += json.dumps(value, ensure_ascii=False)
        piece = ("{" if self._n_params == 0 else ", ") + piece
        self._n_params += 1
        out.calls.append(ToolCallDelta(self._call.index, arguments=piece))

    def _close_call(self, out: ToolDelta) -> None:
        if self._call is None:
            return
        piece = "{}" if self._n_params == 0 else "}"
        out.calls.append(ToolCallDelta(self._call.index, arguments=piece))
        self._call = None

    # -- the state machine --

    def _consume(self, n: int) -> None:
        self._buf = self._buf[n:]

    def _step(self, out: ToolDelta, final: bool) -> bool:
        """Consume what can be decided; ``True`` to call again. ``final``: no more text."""
        s = self._state
        if s is _State.TEXT:
            i, tag = find_tag(self._buf, (TOOL_START, FUNC_START))
            out.content += self._content.text(self._buf[:i])
            self._consume(i)
            if tag == TOOL_START:
                self._content.call()
                self._consume(len(tag))
                self._state = _State.CALL
            elif tag == FUNC_START:
                self._consume(len(tag))
                self._state = _State.NAME
            return tag is not None

        if s in (_State.CALL, _State.AFTER):
            # Only whitespace belongs here; anything else is dropped as malformed.
            i, tag = find_tag(self._buf, (FUNC_START, TOOL_END, TOOL_START))
            self._consume(i)
            if tag is None:
                return False
            self._consume(len(tag))
            self._state = {FUNC_START: _State.NAME, TOOL_END: _State.TEXT}.get(tag, _State.CALL)
            return True

        if s is _State.NAME:
            i = self._buf.find(">")
            if i == -1:
                return False  # at the end, an unfinished name is not a call
            self._open_call(self._buf[:i], out)
            self._consume(i + 1)
            self._state = _State.ARGS
            return True

        if s is _State.ARGS:
            tags = (PARAM_START, FUNC_END, TOOL_END, TOOL_START, FUNC_START)
            i, tag = find_tag(self._buf, tags)
            self._consume(i)
            if tag is None:
                return False
            self._consume(len(tag))
            if tag == PARAM_START:
                self._state = _State.PARAM
                return True
            self._close_call(out)  # </function>, or a call that was never closed
            self._state = {
                FUNC_END: _State.AFTER,
                TOOL_END: _State.TEXT,
                TOOL_START: _State.CALL,
                FUNC_START: _State.NAME,
            }[tag]
            return True

        # PARAM: "KEY>VALUE" up to its end.
        gt = self._buf.find(">")
        if gt != -1:
            i, tag = find_tag(self._buf[gt + 1 :], (PARAM_END, PARAM_START, FUNC_END, TOOL_END))
            if tag is not None or final:
                end = gt + 1 + (i if tag is not None else len(self._buf) - gt - 1)
                self._add_param(self._buf[:gt], self._buf[gt + 1 : end], out)
                self._consume(end + (len(tag) if tag == PARAM_END else 0))
                self._state = _State.ARGS
                return tag is not None
        elif final:
            self._buf = ""  # an unfinished parameter name
        return False

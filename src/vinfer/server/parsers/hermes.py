"""Tool calls as JSON in ``<tool_call>`` tags ("hermes"), the format of Qwen3 and Qwen2.5::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Hanoi"}}
    </tool_call>

A call is sent whole once its closing tag arrives (the JSON is not streamed piece by piece).
A block that is not a valid call stays in the content as it was written.
"""

from __future__ import annotations

import json

from vinfer.server.parsers.base import ToolCallParser, ToolDelta, find_tag
from vinfer.server.parsers.registry import TOOL_CALL_PARSERS
from vinfer.server.parsers.tool_text import ContentAround

TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"


@TOOL_CALL_PARSERS.register("hermes")
class HermesToolCallParser(ToolCallParser):
    start_tags = (TOOL_START,)

    def __init__(self, tools=()) -> None:
        super().__init__(tools)
        self._buf = ""
        self._in_call = False
        self._content = ContentAround()

    def feed(self, text: str) -> ToolDelta:
        self._buf += text
        out = ToolDelta()
        while self._step(out):
            pass
        return out

    def finish(self) -> ToolDelta:
        out = ToolDelta()
        if self._in_call:
            self._emit_call(self._buf, out, closed=False)
        else:
            out.content += self._content.text(self._buf)
        self._buf = ""
        out.content += self._content.finish()
        return out

    def _step(self, out: ToolDelta) -> bool:
        if not self._in_call:
            i, tag = find_tag(self._buf, (TOOL_START,))
            out.content += self._content.text(self._buf[:i])
            self._buf = self._buf[i:]
            if tag is None:
                return False
            self._buf = self._buf[len(tag) :]
            self._in_call = True
            return True
        i = self._buf.find(TOOL_END)
        if i == -1:
            return False
        self._emit_call(self._buf[:i], out, closed=True)
        self._buf = self._buf[i + len(TOOL_END) :]
        self._in_call = False
        return True

    def _emit_call(self, body: str, out: ToolDelta, closed: bool) -> None:
        try:
            obj = json.loads(body)
        except ValueError:
            obj = None
        args = obj.get("arguments", obj.get("parameters", {})) if isinstance(obj, dict) else None
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            raw = TOOL_START + body + (TOOL_END if closed else "")
            out.content += self._content.text(raw)
            return
        if isinstance(args, str):  # some models write the arguments as a JSON string
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        self._content.call()
        call = self._new_call(obj["name"])
        call.arguments = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)
        out.calls.append(call)

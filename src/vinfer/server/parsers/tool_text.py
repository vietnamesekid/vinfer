"""What tool-call parsers share: the content around the calls, and typing parameter values."""

from __future__ import annotations

import json
import math
from typing import Any


class ContentAround:
    """Content outside tool calls. Whitespace right before or after a call is markup (the
    templates put newlines there): a trailing whitespace run is held until real text follows,
    and a call drops it, as does the end of an output that had calls; whitespace that follows a
    call is dropped."""

    def __init__(self) -> None:
        self.held = ""
        self.had_call = False
        self.after_call = False

    def text(self, text: str) -> str:
        if self.after_call:
            text = text.lstrip()
            self.after_call = not text
        if not text:
            return ""
        stripped = text.rstrip()
        if not stripped:
            self.held += text
            return ""
        out = self.held + stripped
        self.held = text[len(stripped) :]
        return out

    def call(self) -> None:
        self.held, self.had_call, self.after_call = "", True, True

    def finish(self) -> str:
        out = "" if self.had_call else self.held
        self.held = ""
        return out


def _schema_types(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return []
    t = schema.get("type")
    types = [t] if isinstance(t, str) else [x for x in t if isinstance(x, str)] if t else []
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            types += _schema_types(sub)
    if not types and "enum" in schema:
        types = ["string"]
    return [x.lower() for x in types]


def coerce_value(value: str, schema: Any) -> Any:
    """A parameter written as text, typed by its JSON schema. Types are tried in the order
    null, integer, number, boolean, object, array, string (vLLM's ``coerce_to_schema_type``);
    a value that fits none stays a string. Without a schema the value is a string."""
    types = _schema_types(schema)
    if not types:
        return value
    v = value.strip()
    for t in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if t not in types:
            continue
        if t == "null" and v.lower() == "null":
            return None
        if t == "integer":
            try:
                return int(v)
            except ValueError:
                pass
        if t == "number":
            try:
                f = float(v)
            except ValueError:
                continue
            if math.isfinite(f):
                return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
        if t == "boolean" and v.lower() in ("true", "false"):
            return v.lower() == "true"
        if t in ("object", "array"):
            try:
                parsed = json.loads(v)
            except ValueError:
                continue
            if isinstance(parsed, dict if t == "object" else list):
                return parsed
        if t == "string":
            return value
    return value

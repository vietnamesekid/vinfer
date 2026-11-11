"""Tiny OpenAI-compatible streaming server with fixed delays, for testing the harness on CPU.

Each request waits ``prefill_s`` before the first token, then ``decode_s`` per token, and
produces exactly ``max_tokens`` tokens. With these known delays the harness metrics have known
expected values (TTFT ~ prefill_s, TPOT ~ decode_s). Stdlib only.

Usage: ``python -m bench.harness fake-server --port 8001``
"""

from __future__ import annotations

import asyncio
import json


class FakeServer:
    def __init__(
        self,
        prefill_s: float = 0.02,
        decode_s: float = 0.005,
        short_by: int = 0,  # tests: generate this many tokens fewer than asked
        error_after: int | None = None,  # tests: send an error event after N tokens
    ) -> None:
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self.short_by = short_by
        self.error_after = error_after
        self.requests_seen = 0
        self.cache_resets = 0
        self.prompts: list[str] = []  # content of the last message of every request
        self.events: list[str] = []  # "request" / "reset", in arrival order
        self.max_inflight = 0
        self._inflight = 0
        self._server: asyncio.base_events.Server | None = None
        self.port: int | None = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:  # keep-alive loop
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode().split("\r\n")
                method, path, _ = lines[0].split(" ", 2)
                headers = {
                    k.lower(): v.strip()
                    for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)
                }
                body = await reader.readexactly(int(headers.get("content-length", 0)))
                if method == "GET" and path in ("/health", "/v1/models"):
                    msg = b'{"ok":true}'
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        b"Content-Length: %d\r\n\r\n%s" % (len(msg), msg)
                    )
                    await writer.drain()
                    continue
                if method == "POST" and path == "/reset_prefix_cache":
                    self.cache_resets += 1
                    self.events.append("reset")
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                    await writer.drain()
                    continue
                if method != "POST" or path != "/v1/chat/completions":
                    writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                    await writer.drain()
                    continue
                await self._stream(json.loads(body), writer)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    async def _stream(self, req: dict, writer: asyncio.StreamWriter) -> None:
        self.requests_seen += 1
        self.prompts.append(req["messages"][-1]["content"])
        self.events.append("request")
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        n = int(req.get("max_tokens") or 16) - self.short_by
        prompt_tokens = sum(len(m["content"].split()) for m in req["messages"])
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )

        async def send(obj: object) -> None:
            data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode())
            data += b"\n\n"
            writer.write(b"%x\r\n%s\r\n" % (len(data), data))
            await writer.drain()

        try:
            await send({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
            await asyncio.sleep(self.prefill_s)
            for i in range(n):
                if i:
                    await asyncio.sleep(self.decode_s)
                if i == self.error_after:
                    await send({"error": {"message": "fake failure", "code": 500}})
                    break
                await send({"choices": [{"index": 0, "delta": {"content": f" t{i}"}}]})
            await send(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": n,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                }
            )
            await send(b"[DONE]")
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            self._inflight -= 1

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self._server = await asyncio.start_server(self._handle, host, port)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


async def serve_forever(port: int, prefill_s: float, decode_s: float) -> None:
    server = FakeServer(prefill_s, decode_s)
    port = await server.start(port=port)
    print(f"fake server on http://127.0.0.1:{port}", flush=True)
    await asyncio.Event().wait()

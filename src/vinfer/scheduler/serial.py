"""The simplest scheduler: one request at a time, in arrival order (no batching).

Generation runs on a single worker thread, so requests are served strictly one after another
and the event loop stays free for HTTP. Each request streams its steps through its own queue.
A consumer that stops reading (client gone) cancels its request at the next token.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor

from vinfer.runtime import SamplingParams, Step, generate
from vinfer.state import StatePool

_DONE = object()


class SerialScheduler:
    def __init__(
        self,
        model,
        pool: StatePool,
        eos_token_ids: frozenset[int] = frozenset(),
        decode: Callable[[list[int]], str] | None = None,
    ) -> None:
        self.model = model
        self.pool = pool
        self.eos_token_ids = eos_token_ids
        self.decode = decode
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vinfer-gen")

    async def stream(self, prompt_ids: list[int], params: SamplingParams) -> AsyncIterator[Step]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()

        def run() -> None:
            try:
                steps = generate(
                    self.model, self.pool, prompt_ids, params, self.eos_token_ids, self.decode
                )
                for step in steps:
                    loop.call_soon_threadsafe(queue.put_nowait, step)
                    if cancelled.is_set():
                        steps.close()  # frees the slot
                        break
            except Exception as exc:  # surfaced to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        future = loop.run_in_executor(self._worker, run)
        try:
            while (item := await queue.get()) is not _DONE:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancelled.set()
            await asyncio.shield(future)

    def close(self) -> None:
        self._worker.shutdown(wait=True)

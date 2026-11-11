"""Peak GPU memory during a run, sampled with NVML from a background thread.

NVML reports device memory used by *all* processes. For vLLM/SGLang this is mostly the memory
they reserve up front (``gpu_memory_utilization`` / ``mem_fraction_static``), not what the
workload actually needs. Read peak VRAM together with the engine config.
Returns ``None`` when NVML is unavailable (e.g. on a laptop), never fails the run.
"""

from __future__ import annotations

import contextlib
import threading
from types import TracebackType


class PeakGpuMemory:
    """Context manager: ``with PeakGpuMemory() as mem: ...``, then ``mem.peak_gb``."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handles: list = []
        self._nvml = None

    def __enter__(self) -> PeakGpuMemory:
        try:
            import pynvml

            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            self._nvml = pynvml
        except Exception:
            return self
        self.peak_bytes = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        """Update the peak; a failed sample (transient NVML error) keeps the last peak."""
        try:
            used = sum(self._nvml.nvmlDeviceGetMemoryInfo(h).used for h in self._handles)
        except Exception:
            return
        self.peak_bytes = max(self.peak_bytes or 0, used)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._sample()
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()

    @property
    def peak_gb(self) -> float | None:
        return None if self.peak_bytes is None else self.peak_bytes / 1e9

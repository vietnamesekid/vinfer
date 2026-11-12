"""The Gated DeltaNet backend interface and its registry.

State is updated **in place**: the backend receives views into the state pool and leaves the
new state there, as the fused FLA kernels do. A backend that cannot work in place copies
internally; callers never see the difference.

Shapes are for one sequence. Batched (varlen) entry points with per-sequence slot indices come
with continuous batching, as new methods of this interface.
"""

from __future__ import annotations

from typing import Protocol

import torch

from vinfer.kernels.registry import BackendRegistry


class GDNBackend(Protocol):
    def conv(self, x: torch.Tensor, weight: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        """SiLU(depthwise causal conv) of ``x`` ``[T, C]`` continuing from ``conv_state``
        ``[C, kernel-1]`` (updated in place to the last raw inputs). ``weight`` ``[C, kernel]``.
        Returns ``[T, C]`` in ``x``'s dtype."""
        ...

    def rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> torch.Tensor:
        """Gated delta rule over ``T`` tokens, starting from ``ssm_state`` ``[HV, V, K]``
        (updated in place). ``q``, ``k`` ``[T, H, K]`` (L2-normalized inside); ``v``
        ``[T, HV, V]``; ``g``, ``beta`` ``[T, HV]`` fp32. Returns ``[T, HV, V]`` in ``v``'s
        dtype. The backend picks its own algorithm (e.g. recurrent for decode, chunked for
        prefill); all must agree within the parity thresholds."""
        ...


GDN: BackendRegistry[GDNBackend] = BackendRegistry("gdn")

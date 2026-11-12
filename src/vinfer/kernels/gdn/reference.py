"""Slow, readable Gated DeltaNet (the linear-attention layer of Qwen3.5/3.6). Ground truth for
every faster GDN kernel.

Per value head, with recurrent state ``S`` of shape ``[V, K]`` and inputs at step ``t``:

    S   = S * exp(g_t)                          # decay (g_t <= 0, log space)
    S   = S + beta_t * (v_t - S k_t) ⊗ k_t      # delta rule: write only what S does not predict
    o_t = S q_t

``q`` and ``k`` are L2-normalized and ``q`` is scaled by ``K^-0.5``. When there are more value
heads than key heads (``HV = r * H``), value head ``j`` reads key head ``j // r``.

Two forms compute the same recurrence:

- ``recurrent_gated_delta_rule``: one step per token. The definition; used for decode.
- ``chunk_gated_delta_rule``: blocks of ``chunk_size`` tokens, each block solved with the UT
  transform (a unit-lower-triangular solve), the state carried between blocks. Used for prefill.
  The algorithm follows ``torch_chunk_gated_delta_rule`` in HF Transformers
  (src/transformers/models/qwen3_5/modeling_qwen3_5.py, Apache 2.0), rewritten for the state
  layout used here.

Layout and dtypes (one convention for the whole engine):

- ``q``, ``k``: ``[B, T, H, K]``; ``v``: ``[B, T, HV, V]``; ``g``, ``beta``: ``[B, T, HV]``.
- Recurrent state: ``[B, HV, V, K]``, fp32. HF Transformers stores ``[B, HV, K, V]``; the two
  are transposes of each other, which goes unnoticed when ``K == V`` (as in Qwen3.5), so tests
  use ``K != V``.
- Everything is computed in fp32; the output is returned in the dtype of ``v``.
- Short convolution state: the last ``kernel - 1`` **raw** inputs (before conv and SiLU),
  ``[B, C, kernel - 1]``, in the dtype of the activations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vinfer.kernels.gdn.base import GDN
from vinfer.kernels.registry import BackendContext


def gdn_gating(
    a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log decay ``g = -exp(A_log) * softplus(a + dt_bias)`` and write strength
    ``beta = sigmoid(b)``, both fp32, shape of ``a``/``b`` (``[..., HV]``).

    HF Transformers takes ``sigmoid`` in the activation dtype; vLLM, SGLang and this reference
    take it in fp32 (``docs/models/qwen3.5.md``).
    """
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g, b.float().sigmoid()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x * rsqrt(sum(x²) + eps)`` over the last dim (eps inside the square root, as FLA)."""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, ...]:
    """fp32 copies in head-major layout ``[B, HV, T, D]``, q/k repeated to ``HV`` heads."""
    batch, _, num_k_heads, k_dim = k.shape
    num_v_heads, v_dim = v.shape[-2:]
    if num_v_heads % num_k_heads:
        raise ValueError(f"value heads {num_v_heads} not a multiple of key heads {num_k_heads}")
    ratio = num_v_heads // num_k_heads
    q, k = (x.float().repeat_interleave(ratio, dim=2) for x in (q, k))
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    q = q * k_dim**-0.5
    q, k, v = (x.transpose(1, 2).float() for x in (q, k, v))  # [B, HV, T, D]
    g, beta = (x.transpose(1, 2).float() for x in (g, beta))  # [B, HV, T]
    if initial_state is None:
        state = torch.zeros(batch, num_v_heads, v_dim, k_dim, dtype=torch.float32, device=v.device)
    else:
        expected = (batch, num_v_heads, v_dim, k_dim)
        if tuple(initial_state.shape) != expected:
            raise ValueError(f"state shape {tuple(initial_state.shape)}, expected {expected}")
        state = initial_state.float().clone()
    return q, k, v, g, beta, state


def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token recurrence. Returns ``(o [B, T, HV, V] in v's dtype, state [B, HV, V, K]
    fp32)``. ``initial_state`` is not modified."""
    out_dtype = v.dtype
    q, k, v, g, beta, state = _prepare(q, k, v, g, beta, initial_state, use_qk_l2norm)
    out = torch.empty_like(v)
    for t in range(v.shape[2]):
        state = state * g[:, :, t].exp()[..., None, None]
        predicted = (state * k[:, :, t, None, :]).sum(-1)  # S k: [B, HV, V]
        delta = (v[:, :, t] - predicted) * beta[:, :, t, None]
        state = state + delta[..., :, None] * k[:, :, t, None, :]
        out[:, :, t] = (state * q[:, :, t, None, :]).sum(-1)  # S q
    return out.transpose(1, 2).to(out_dtype), state


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same result as ``recurrent_gated_delta_rule`` (up to fp32 rounding), in blocks.

    Padding the last block is neutral: padded steps have ``k = v = beta = 0`` (no write) and
    ``g = 0`` (no decay), so the final state is unchanged by them.
    """
    out_dtype = v.dtype
    q, k, v, g, beta, state = _prepare(q, k, v, g, beta, initial_state, use_qk_l2norm)
    seq_len = v.shape[2]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    g, beta = (F.pad(x, (0, pad)) for x in (g, beta))

    def blocks(x: torch.Tensor) -> torch.Tensor:  # [B, HV, T, ...] -> [B, HV, N, C, ...]
        return x.reshape(*x.shape[:2], -1, chunk_size, *x.shape[3:])

    q, k, v, g, beta = (blocks(x) for x in (q, k, v, g, beta))
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    cum_g = g.cumsum(-1)  # log decay from the block start to each position (inclusive)
    future = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=v.device).triu(1)
    # decay[i, j] = exp(cum_g[i] - cum_g[j]) for j <= i: decay accumulated from step j to step i.
    decay = (cum_g[..., :, None] - cum_g[..., None, :]).masked_fill(future, float("-inf")).exp()

    # UT transform: within a block, the delta-rule writes form a unit-lower-triangular system.
    # T = I + strictly_lower(beta_i k_i·k_j decay[i, j]); new values u = T^-1 v_beta, and the
    # keys that read the block-start state w = T^-1 (k_beta * exp(cum_g)).
    eye = torch.eye(chunk_size, dtype=torch.float32, device=v.device)
    system = ((k_beta @ k.transpose(-1, -2)) * decay).tril(-1) + eye
    u = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(
        system, k_beta * cum_g.exp()[..., None], upper=False, unitriangular=True
    )
    intra = (q @ k.transpose(-1, -2)) * decay  # within-block attention, causal incl. diagonal
    q_decayed = q * cum_g.exp()[..., None]  # reads the block-start state
    k_to_end = k * (cum_g[..., -1:] - cum_g).exp()[..., None]  # decay from step to block end
    block_decay = cum_g[..., -1].exp()  # [B, HV, N]

    out = torch.empty_like(v)
    for n in range(v.shape[2]):
        # state is [B, HV, V, K]; S^T as [K, V] makes the matmuls read like the equations.
        s_kv = state.transpose(-1, -2)
        u_n = u[:, :, n] - w[:, :, n] @ s_kv  # correction not predicted by the old state
        out[:, :, n] = q_decayed[:, :, n] @ s_kv + intra[:, :, n] @ u_n
        s_kv = s_kv * block_decay[:, :, n, None, None] + k_to_end[:, :, n].transpose(-1, -2) @ u_n
        state = s_kv.transpose(-1, -2)
    out = out.reshape(*out.shape[:2], -1, out.shape[-1])[:, :, :seq_len]
    return out.transpose(1, 2).to(out_dtype), state.contiguous()


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor | None = None,
    activation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution over time, continuing from ``conv_state``.

    Args:
        x: ``[B, T, C]``.
        weight: ``[C, kernel]`` (the checkpoint's ``[C, 1, kernel]`` squeezed).
        conv_state: ``[B, C, kernel - 1]`` raw inputs preceding ``x``; zeros if ``None``.
        activation: apply SiLU to the output (Qwen3.5 does).

    Returns:
        ``(y [B, T, C], new_conv_state [B, C, kernel - 1])``. The computation runs in the dtype
        of ``weight``, like HF Transformers; ``y`` is returned in the dtype of ``x``.
    """
    batch, seq_len, channels = x.shape
    width = weight.shape[-1]
    if conv_state is None:
        conv_state = x.new_zeros(batch, channels, width - 1)
    full = torch.cat([conv_state.to(x.dtype), x.transpose(1, 2)], dim=-1)  # [B, C, W-1+T]
    if seq_len == 0:
        return x, full.contiguous()
    y = F.conv1d(full.to(weight.dtype), weight[:, None, :], groups=channels)  # [B, C, T]
    if activation:
        y = F.silu(y)
    return y.transpose(1, 2).to(x.dtype), full[..., -(width - 1) :].contiguous()


@GDN.register("reference")
class ReferenceGDN:
    """``GDNBackend`` over the functions above: recurrent form for one token, chunked form for
    more; any device/dtype. Works on copies and writes the new state back (not fused)."""

    def __init__(self, ctx: BackendContext) -> None:
        self.ctx = ctx

    def conv(self, x: torch.Tensor, weight: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        y, new_state = causal_conv1d(x[None], weight, conv_state[None])
        conv_state.copy_(new_state[0])
        return y[0]

    def rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> torch.Tensor:
        form = recurrent_gated_delta_rule if v.shape[0] == 1 else chunk_gated_delta_rule
        out, final = form(q[None], k[None], v[None], g[None], beta[None], ssm_state[None])
        ssm_state.copy_(final[0])
        return out[0]

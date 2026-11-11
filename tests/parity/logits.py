"""Compact per-prompt logit summaries, and the metrics that compare two of them.

Full logits are too big to keep for every position (vocab 248,320 x 4 bytes ~ 1 MB per
position), so a ``LogitSummary`` keeps, for each position, the argmax, the top-k logits and the
log-sum-exp, and full logits only at a few sampled positions (the last ones, where decoding
starts, plus random ones). KL and max |diff| are computed on those full positions; argmax
agreement on every position.

Logits are produced by applying the LM head to the final hidden states in chunks of positions,
so a 32k-token prompt never materialises a [32k, vocab] tensor. Reference and candidate must go
through the same ``summarize`` for the comparison to be fair.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOPK = 20


@dataclass
class LogitSummary:
    argmax: torch.Tensor  # [seq] int64
    topk_ids: torch.Tensor  # [seq, k] int64
    topk_vals: torch.Tensor  # [seq, k] float32 (raw logits)
    lse: torch.Tensor  # [seq] float32
    full_pos: torch.Tensor  # [n] int64
    full: torch.Tensor  # [n, vocab] float32

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({f.name: getattr(self, f.name).contiguous() for f in fields(self)}, str(path))

    @classmethod
    def load(cls, path: Path) -> LogitSummary:
        return cls(**load_file(str(path)))


def full_positions(seq_len: int, n_last: int = 16, n_random: int = 16, seed: int = 0) -> list[int]:
    """Sorted positions whose full logits are kept: the last ``n_last`` plus random others."""
    last = set(range(max(0, seq_len - n_last), seq_len))
    rest = [p for p in range(seq_len) if p not in last]
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(rest), generator=g)[:n_random].tolist()
    return sorted(last | {rest[i] for i in pick})


def summarize(chunks: Iterable[torch.Tensor], full_pos: list[int], k: int = TOPK) -> LogitSummary:
    """Summarise logits given as consecutive ``[chunk, vocab]`` blocks covering every position."""
    want = set(full_pos)
    argmax, tid, tval, lse, full = [], [], [], [], []
    start = 0
    for c in chunks:
        c = c.float()
        argmax.append(c.argmax(-1).cpu())
        v, i = c.topk(k, dim=-1)
        tid.append(i.cpu())
        tval.append(v.cpu())
        lse.append(torch.logsumexp(c, -1).cpu())
        for j in range(c.shape[0]):
            if start + j in want:
                full.append(c[j].cpu())
        start += c.shape[0]
    return LogitSummary(
        argmax=torch.cat(argmax),
        topk_ids=torch.cat(tid),
        topk_vals=torch.cat(tval),
        lse=torch.cat(lse),
        full_pos=torch.tensor(sorted(want), dtype=torch.int64),
        full=torch.stack(full) if full else torch.empty(0),
    )


def compare(ref: LogitSummary, cand: LogitSummary) -> dict[str, float]:
    """Metrics of ``cand`` against ``ref`` for one prompt."""
    if ref.argmax.shape != cand.argmax.shape:
        raise ValueError(f"length mismatch {ref.argmax.shape} vs {cand.argmax.shape}")
    if not torch.equal(ref.full_pos, cand.full_pos):
        raise ValueError("full_pos differ: summaries built with different position samples")
    diff = (ref.full - cand.full).abs()
    lp_ref = torch.log_softmax(ref.full, -1)
    lp_cand = torch.log_softmax(cand.full, -1)
    kl = (lp_ref.exp() * (lp_ref - lp_cand)).sum(-1)  # KL(ref || cand) per position
    mism = ref.argmax != cand.argmax
    return {
        "positions": int(ref.argmax.numel()),
        "argmax_mismatch": int(mism.sum()),
        "argmax_mismatch_rate": float(mism.float().mean()),
        "first_mismatch_pos": int(mism.nonzero()[0, 0]) if mism.any() else -1,
        "max_abs_logit": float(diff.max()),
        "mean_abs_logit": float(diff.mean()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "lse_max_abs": float((ref.lse - cand.lse).abs().max()),
        "bitwise_equal": bool(torch.equal(ref.full, cand.full) and not mism.any()),
    }


def aggregate(per_prompt: dict[str, dict[str, float]]) -> dict[str, float]:
    """Worst case over prompts (max) for diffs, totals for counts."""
    vals = list(per_prompt.values())
    pos = sum(v["positions"] for v in vals)
    mis = sum(v["argmax_mismatch"] for v in vals)
    return {
        "prompts": len(vals),
        "positions": pos,
        "argmax_mismatch": mis,
        "argmax_mismatch_rate": mis / pos if pos else 0.0,
        "max_abs_logit": max(v["max_abs_logit"] for v in vals),
        "mean_abs_logit": sum(v["mean_abs_logit"] for v in vals) / len(vals),
        "kl_mean": sum(v["kl_mean"] for v in vals) / len(vals),
        "kl_max": max(v["kl_max"] for v in vals),
        "bitwise_equal": all(v["bitwise_equal"] for v in vals),
    }

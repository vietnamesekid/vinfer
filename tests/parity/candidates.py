"""Things whose logits can be compared: HF Transformers (the reference) and the vinfer engine.

A candidate turns token ids into ``LogitSummary`` objects (``tests/parity/logits.py``).
HF knobs exposed here are exactly the noise-floor dimensions (``noise_floor.py``): dtype,
batch size, padding side, request order, attention implementation.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from tests.parity.logits import LogitSummary, full_positions, summarize

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class Candidate(Protocol):
    name: str

    def summaries(self, prompts: Sequence[list[int]]) -> list[LogitSummary]: ...


@dataclass
class HFConfig:
    model: str
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    batch_size: int = 1
    padding_side: Literal["left", "right"] = "right"
    order_seed: int | None = None  # None = prompt order; int = shuffled order
    chunk: int = 1024  # positions per LM-head chunk

    @property
    def name(self) -> str:
        return (
            f"hf[{self.dtype},{self.attn_implementation},bs={self.batch_size},"
            f"pad={self.padding_side},order={self.order_seed}]"
        )


def environment() -> dict[str, str]:
    """Everything that can change HF logits bit-for-bit besides the model and dtype."""
    import transformers

    return {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gdn_backend": gdn_backend(),
        "conv_backend": conv_backend(),
        "flash-linear-attention": _dist_version("flash-linear-attention"),
        "causal-conv1d": _dist_version("causal-conv1d"),
    }


# Optional kernel libraries HF Transformers uses for Gated DeltaNet layers when importable,
# falling back to PyTorch otherwise. Each one changes the logits, so each is recorded.
def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _dist_version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "none"


def gdn_backend() -> str:
    """Gated DeltaNet recurrence (chunk_gated_delta_rule): fla kernels if importable."""
    return "fla" if _installed("fla") else "torch"


def conv_backend() -> str:
    """Gated DeltaNet short convolution (causal_conv1d_fn): the CUDA kernel if importable."""
    return "causal_conv1d" if _installed("causal_conv1d") else "torch"


def backend_tag() -> str:
    """Short label of the kernel combination, for file names: torch, fla, fla-cconv, ..."""
    return gdn_backend() + ("-cconv" if conv_backend() == "causal_conv1d" else "")


class HFCandidate:
    def __init__(self, cfg: HFConfig, model: torch.nn.Module | None = None) -> None:
        self.cfg = cfg
        self.name = cfg.name
        if model is None:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                revision=cfg.revision,
                dtype=DTYPES[cfg.dtype],
                attn_implementation=cfg.attn_implementation,
                device_map=cfg.device,  # load straight to the device, no CPU copy first
            )
        self.model = model.eval()

    def _chunks(self, hidden: torch.Tensor) -> Iterator[torch.Tensor]:
        for s in range(0, hidden.shape[0], self.cfg.chunk):
            yield self.model.lm_head(hidden[s : s + self.cfg.chunk])

    @torch.inference_mode()
    def summaries(self, prompts: Sequence[list[int]]) -> list[LogitSummary]:
        order = list(range(len(prompts)))
        if self.cfg.order_seed is not None:
            g = torch.Generator().manual_seed(self.cfg.order_seed)
            order = torch.randperm(len(prompts), generator=g).tolist()
        out: dict[int, LogitSummary] = {}
        bs = self.cfg.batch_size
        dev = next(self.model.parameters()).device
        for b in range(0, len(order), bs):
            idx = order[b : b + bs]
            seqs = [prompts[i] for i in idx]
            width = max(len(s) for s in seqs)
            ids = torch.zeros(len(seqs), width, dtype=torch.long)
            mask = torch.zeros(len(seqs), width, dtype=torch.long)
            for r, s in enumerate(seqs):
                if self.cfg.padding_side == "right":
                    ids[r, : len(s)] = torch.tensor(s)
                    mask[r, : len(s)] = 1
                else:
                    ids[r, width - len(s) :] = torch.tensor(s)
                    mask[r, width - len(s) :] = 1
            kwargs = {"input_ids": ids.to(dev)}
            if len(seqs) > 1:
                kwargs["attention_mask"] = mask.to(dev)
                # Positions count real tokens only, as HF generate does for left padding.
                kwargs["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0).to(dev)
            # use_cache=False: no KV / recurrent-state cache is needed for one forward pass.
            hidden = self.model.model(**kwargs, use_cache=False).last_hidden_state
            for r, (i, s) in enumerate(zip(idx, seqs, strict=True)):
                h = (
                    hidden[r, : len(s)]
                    if self.cfg.padding_side == "right"
                    else hidden[r, -len(s) :]
                )
                out[i] = summarize(self._chunks(h), full_positions(len(s)))
        return [out[i] for i in range(len(prompts))]


class VinferCandidate:
    """The vinfer engine, one prompt at a time (batch size 1).

    Long prompts are prefilled in pieces of ``prefill_chunk`` tokens: the reference attention
    materializes ``[heads, T, T]`` scores, which for a 16K-token prompt would take tens of GB.
    Prefilling in pieces gives the same result (tests/unit/models/test_qwen_model.py).
    """

    def __init__(
        self,
        path: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        prefill_chunk: int = 2048,
        lm_head_chunk: int = 1024,
        model=None,
    ) -> None:
        from vinfer.models.qwen import load_qwen

        self.model = model if model is not None else load_qwen(path, DTYPES[dtype], device)
        self.prefill_chunk = prefill_chunk
        self.lm_head_chunk = lm_head_chunk
        self.name = f"vinfer[{dtype},prefill_chunk={prefill_chunk}]"

    def hidden(self, ids: list[int], layer_outputs: list | None = None) -> torch.Tensor:
        model = self.model
        device = model.embed_tokens.weight.device
        pool = model.new_state_pool(num_slots=1, max_seq_len=len(ids))
        slot = pool.alloc()
        pieces = []
        for start in range(0, len(ids), self.prefill_chunk):
            piece = torch.tensor(ids[start : start + self.prefill_chunk], device=device)
            pieces.append(model.forward_hidden(piece, start, slot, pool, layer_outputs))
        return torch.cat(pieces)

    @torch.inference_mode()
    def summaries(self, prompts: Sequence[list[int]]) -> list[LogitSummary]:
        out = []
        for ids in prompts:
            hidden = self.hidden(list(ids))
            chunks = (
                self.model.logits(hidden[s : s + self.lm_head_chunk])
                for s in range(0, hidden.shape[0], self.lm_head_chunk)
            )
            out.append(summarize(chunks, full_positions(len(ids))))
        return out

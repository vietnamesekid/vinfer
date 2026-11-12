"""Find where vinfer and HF Transformers diverge, layer by layer, on one prompt.

For debugging a parity failure: the first layer whose output moves away from HF (relative to
the size of HF's output) is where to look.

Usage (GPU)::

    python -m tests.parity.layer_diff --model Qwen/Qwen3.5-2B --prompt short-en-0 [--dtype bfloat16]
"""

from __future__ import annotations

import argparse

import torch

from tests.parity.candidates import DTYPES, VinferCandidate


def _decoder_layers(hf_model: torch.nn.Module) -> torch.nn.ModuleList:
    for name, module in hf_model.named_modules():
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList):
            return module
    raise ValueError("no decoder layer list found in the HF model")


@torch.inference_mode()
def layer_diffs(hf_model: torch.nn.Module, vinfer_model, ids: list[int]) -> list[dict]:
    """Per decoder layer (then the logits): max and mean |vinfer - HF|, and max relative to
    HF's largest value. One forward each, the whole prompt at once."""
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append((output[0] if isinstance(output, tuple) else output)[0].float())

    layers = _decoder_layers(hf_model)
    handles = [layer.register_forward_hook(hook) for layer in layers]
    try:
        device = next(hf_model.parameters()).device
        hf_logits = hf_model(torch.tensor([ids], device=device)).logits[0].float()
    finally:
        for h in handles:
            h.remove()

    ours: list[torch.Tensor] = []
    candidate = VinferCandidate("", model=vinfer_model, prefill_chunk=len(ids))
    hidden = candidate.hidden(ids, ours)
    our_logits = vinfer_model.logits(hidden).float()

    rows = []
    pairs = [*zip(ours, captured, strict=True), (our_logits, hf_logits)]
    for i, (a, b) in enumerate(pairs):
        diff = (a.float() - b.to(a.device)).abs()
        rows.append(
            {
                "layer": i if i < len(captured) else "logits",
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "max_rel": float(diff.max() / b.abs().max().clamp_min(1e-12)),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--revision", help="default: the pin in bench/models.lock")
    p.add_argument("--prompt", required=True, help="prompt id in tests/parity/prompts/v1.jsonl")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.harness.config import pinned_revision
    from tests.parity import prompt_set
    from vinfer.models import load_model

    revision = a.revision or pinned_revision(a.model)
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision)
    prompts = [x for x in prompt_set.load("v1") if x["id"] == a.prompt]
    if not prompts:
        raise SystemExit(f"no prompt {a.prompt!r}")
    ids = prompt_set.tokenize(prompts, tok)[a.prompt]
    hf = AutoModelForCausalLM.from_pretrained(
        a.model, revision=revision, dtype=DTYPES[a.dtype], device_map=a.device
    ).eval()
    path = snapshot_download(a.model, revision=revision, allow_patterns=["*.json", "*.safetensors"])
    ours = load_model(path, DTYPES[a.dtype], a.device)
    print(f"{a.model} @ {revision[:8]}, {a.dtype}, prompt {a.prompt} ({len(ids)} tokens)")
    print(f"{'layer':>7} {'max_abs':>10} {'mean_abs':>10} {'max_rel':>10}")
    for r in layer_diffs(hf, ours, ids):
        print(f"{r['layer']!s:>7} {r['max_abs']:10.4g} {r['mean_abs']:10.4g} {r['max_rel']:10.4g}")


if __name__ == "__main__":
    main()

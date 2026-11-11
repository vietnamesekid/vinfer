"""Download model weights into the Hugging Face cache, at the revisions pinned in
``bench/models.lock``.

Usage (from the repo root): python -m scripts.download_weights [<hf-model-id> ...]
(default: every pinned model)

There is one copy of the weights, in the HF cache (``$HF_HOME``, on the persistent volume on a
rented machine): the baselines (``--revision``), parity and the noise floor all load by repo id
and pinned revision from there. A model that is not pinned is refused: an unpinned download
would silently measure whatever ``main`` is today.
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download

from bench.harness.config import MODELS_LOCK, pinned_revision


def pinned_models() -> list[str]:
    lines = MODELS_LOCK.read_text().splitlines()
    return [line.split()[0] for line in lines if line.strip() and not line.startswith("#")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="*", help="default: every model in bench/models.lock")
    models = p.parse_args().models or pinned_models()
    for model in models:
        revision = pinned_revision(model)
        if revision is None:
            sys.exit(f"{model} is not pinned in bench/models.lock: add `<id> <commit sha>` first")
        path = snapshot_download(model, revision=revision)
        print(f"{model}@{revision[:12]} -> {path}")
    print("Reminder: check each model's license. Weights are not covered by vinfer's license.")


if __name__ == "__main__":
    main()

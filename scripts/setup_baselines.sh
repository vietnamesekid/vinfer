#!/usr/bin/env bash
# Create one isolated venv per baseline engine, pinned by bench/baselines/<engine>/requirements.txt.
# Usage: scripts/setup_baselines.sh [vllm] [sglang]   (default: both)
# Venvs go to $VINFER_VENVS (default ./.venvs; on RunPod setup.sh points it at the volume).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VINFER_VENVS="${VINFER_VENVS:-$ROOT/.venvs}"
engines=("$@"); [[ ${#engines[@]} -eq 0 ]] && engines=(vllm sglang)
for e in "${engines[@]}"; do
  req="$ROOT/bench/baselines/$e/requirements.txt"
  [[ -f "$req" ]] || { echo "unknown engine $e" >&2; exit 1; }
  echo "== $e -> $VINFER_VENVS/$e"
  uv venv --python 3.12 "$VINFER_VENVS/$e"
  VIRTUAL_ENV="$VINFER_VENVS/$e" uv pip install --torch-backend=auto -r "$req"
  VINFER_MODEL=x VINFER_PORT=0 VINFER_VENVS="$VINFER_VENVS" bash "$ROOT/bench/baselines/$e/version.sh"
done

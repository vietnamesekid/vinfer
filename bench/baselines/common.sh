#!/usr/bin/env bash
# Sourced by every launch script bench/baselines/<engine>/configs/<config>.sh.
# Each engine runs from its own venv, because vLLM and SGLang pin torch versions that conflict
# with each other and with vinfer.
# Inputs (set by bench/harness/engines.py): VINFER_MODEL, VINFER_PORT, VINFER_MODEL_REVISION.
set -euo pipefail
ENGINE="$(basename "$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)")"
VINFER_VENVS="${VINFER_VENVS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.venvs}"
VENV="$VINFER_VENVS/$ENGINE"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "missing venv $VENV: run scripts/setup_baselines.sh $ENGINE" >&2
  exit 1
fi
export PATH="$VENV/bin:$PATH"
: "${VINFER_MODEL:?}" "${VINFER_PORT:?}"
REVISION_ARGS=()
if [[ -n "${VINFER_MODEL_REVISION:-}" ]]; then REVISION_ARGS=(--revision "$VINFER_MODEL_REVISION"); fi
# Settings every config of this engine needs for the harness (cache reset, cached-token usage).
HARNESS_ARGS=()
ENGINE_DIR="$(dirname "${BASH_SOURCE[0]}")/$ENGINE"
# shellcheck source=/dev/null
[[ -f "$ENGINE_DIR/harness.sh" ]] && source "$ENGINE_DIR/harness.sh"

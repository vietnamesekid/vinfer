#!/usr/bin/env bash
# Print "<engine> <version> torch <version>" from this engine's venv, or "unknown".
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${VINFER_VENVS:-$ROOT/.venvs}/vllm/bin/python"
"$PY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" 2>/dev/null || echo unknown

#!/usr/bin/env bash
# Set up a freshly rented GPU machine in one command.
#
# RunPod (the standard benchmark machine, bench/HARDWARE.md): Secure Cloud, on-demand,
# 1x H100 80GB, with a network
# volume mounted at /workspace. From a clone of this repo on the pod:
#
#   VINFER_INSTANCE_TYPE="1x H100 80GB SXM" VINFER_DEDICATED=1 ./scripts/setup.sh
#
# Everything slow to rebuild lives on the volume: the HF cache (weights, datasets), baseline
# venvs, the uv cache. Writes .env.bench (KEY=VALUE lines) first, which the Makefile and the
# Python tools read (bench/harness/environment.py:load_env_file); nothing needs sourcing.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------- inputs and early checks
: "${VINFER_INSTANCE_TYPE:?set VINFER_INSTANCE_TYPE, e.g. '1x H100 80GB SXM' (bench/HARDWARE.md)}"
# Results are only official on a dedicated (on-demand, non-spot, non-shared) pod. Say so
# explicitly: 1 = dedicated, 0 = anything else.
: "${VINFER_DEDICATED:?set VINFER_DEDICATED=1 for a dedicated on-demand pod, 0 otherwise}"
VOLUME="${VINFER_VOLUME:-/workspace}"
[[ -d "$VOLUME" ]] || { echo "persistent volume $VOLUME not mounted (set VINFER_VOLUME)" >&2; exit 1; }

# Fail now, not after an hour of downloads, if results could not be traced or pushed.
git rev-parse HEAD >/dev/null 2>&1 || { echo "repo has no commit: clone a pushed commit" >&2; exit 1; }
git remote get-url origin >/dev/null 2>&1 \
  || echo "WARNING: no git remote 'origin': push_results.sh will only back up to the volume" >&2
git config user.email >/dev/null \
  || echo "WARNING: no git identity: set git config user.name/user.email before push_results.sh" >&2

# ---------------------------------------------------------------- .env.bench (written first)
export HF_HOME="${HF_HOME:-$VOLUME/hf}"
export VINFER_VENVS="${VINFER_VENVS:-$VOLUME/venvs}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$VOLUME/uv-cache}"
mkdir -p "$HF_HOME" "$VINFER_VENVS"
cat > .env.bench <<ENV
VINFER_PROVIDER=runpod
VINFER_INSTANCE_TYPE=$VINFER_INSTANCE_TYPE
VINFER_DEDICATED=$VINFER_DEDICATED
HF_HOME=$HF_HOME
VINFER_VENVS=$VINFER_VENVS
UV_CACHE_DIR=$UV_CACHE_DIR
VINFER_PARITY_CACHE=$VOLUME/parity
ENV
echo "== .env.bench written"

# ---------------------------------------------------------------- machine and project env
echo "== GPU / driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi | grep "CUDA Version"

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== project env (from uv.lock)"
uv sync --locked --extra dev --extra parity --extra bench --extra workloads
uv run python - <<'PY'
import platform, torch
assert torch.cuda.is_available(), "torch does not see a GPU"
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

# ---------------------------------------------------------------- baselines
echo "== baseline venvs (skipped if present; delete $VINFER_VENVS/<engine> to rebuild)"
for engine in vllm sglang; do
  [[ -x "$VINFER_VENVS/$engine/bin/python" ]] || scripts/setup_baselines.sh "$engine"
  version="$(bash "bench/baselines/$engine/version.sh")"
  [[ "$version" != unknown ]] || { echo "$engine venv is broken" >&2; exit 1; }
  echo "$version"
done
# ---------------------------------------------------------------- weights and workloads
echo "== weights into the HF cache (revisions pinned in bench/models.lock)"
uv run python -m scripts.download_weights

echo "== workloads (rebuilt from pinned sources; must match the committed manifests)"
uv run python -m bench.workloads.generate

echo "Ready. Before shutting down: scripts/push_results.sh"

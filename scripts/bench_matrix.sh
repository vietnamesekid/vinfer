#!/usr/bin/env bash
# Every baseline engine x launch config, each over every workload, in one rental session.
#
#   MODEL=Qwen/Qwen3.5-2B scripts/bench_matrix.sh
#
# Speed baselines are vLLM and SGLang (HF is the correctness reference only, used by
# `make parity`). A failed combination does not stop the matrix: failures are
# listed at the end and the script exits 1.
# Every config in bench/baselines/<engine>/configs/ runs unless CONFIGS restricts them.
set -uo pipefail
cd "$(dirname "$0")/.."

: "${MODEL:?set MODEL, e.g. Qwen/Qwen3.5-2B}"
ENGINES="${ENGINES:-vllm sglang}"
CONFIGS="${CONFIGS:-}"  # empty = every config of each engine
WORKLOADS="${WORKLOADS:-W1-v1 W2-v1 W3-v1 W4-v1}"
CONCURRENCY="${CONCURRENCY:-1 8 32 128}"
NUM_REQUESTS="${NUM_REQUESTS:-auto}"
RUNS="${RUNS:-3}"

failed=()
for engine in $ENGINES; do
  for script in bench/baselines/"$engine"/configs/*.sh; do
    config="$(basename "$script" .sh)"
    if [[ -n "$CONFIGS" && " $CONFIGS " != *" $config "* ]]; then continue; fi
    # One server launch per engine/config, shared by every workload (the prefix cache is reset
    # before each measured run, so workloads do not see each other).
    echo "=== $engine-$config: $WORKLOADS (concurrency: $CONCURRENCY)"
    # shellcheck disable=SC2086  # word splitting of $WORKLOADS and $CONCURRENCY is intended
    if ! uv run python -m bench.harness run --engine "$engine" --config "$config" \
        --model "$MODEL" --workload $WORKLOADS --concurrency $CONCURRENCY \
        --num-requests "$NUM_REQUESTS" --runs "$RUNS"; then
      failed+=("$engine-$config")
    fi
  done
done

if (( ${#failed[@]} )); then
  printf 'FAILED (see bench/results/raw/<stamp>_<engine>-<config>/server.log):\n'
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "all combinations done"

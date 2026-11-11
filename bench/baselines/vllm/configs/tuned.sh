#!/usr/bin/env bash
# vLLM "tuned": recipe flags for text-only throughput serving.
# Source: https://recipes.vllm.ai/Qwen/Qwen3.6-27B ("Text-only (skip vision encoder), with
# prefix caching") and the Qwen3.5/3.6 guide (text-only throughput command).
# --language-model-only frees the vision encoder memory for KV/state; prefix caching made explicit.
# TODO: sweep --gpu-memory-utilization and --max-num-seqs on the benchmark machine, keep the best,
# and record the sweep in docs/benchmarking.md. Do not change flags here without that record.
source "$(dirname "$0")/../../common.sh"
exec vllm serve "$VINFER_MODEL" "${REVISION_ARGS[@]}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-prefix-caching \
  --seed 0 \
  "${HARNESS_ARGS[@]}" \
  --host 127.0.0.1 --port "$VINFER_PORT"

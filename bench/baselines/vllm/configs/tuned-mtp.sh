#!/usr/bin/env bash
# vLLM "tuned-mtp": tuned + MTP speculative decoding with the model's own MTP head.
# Source: https://recipes.vllm.ai/Qwen/Qwen3.6-27B (MTP command, num_speculative_tokens 1);
# the Qwen3.5/3.6 guide allows 1-5. Use for latency workloads (W1, W3 at low concurrency).
# TODO: try num_speculative_tokens 1/2/3 at concurrency 1 on the benchmark machine, keep the best.
source "$(dirname "$0")/../../common.sh"
exec vllm serve "$VINFER_MODEL" "${REVISION_ARGS[@]}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-prefix-caching \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --seed 0 \
  "${HARNESS_ARGS[@]}" \
  --host 127.0.0.1 --port "$VINFER_PORT"

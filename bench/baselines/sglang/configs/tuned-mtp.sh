#!/usr/bin/env bash
# SGLang "tuned-mtp": tuned + MTP, flags from the cookbook note
# ("To enable MTP ... --speculative-algorithm EAGLE --speculative-num-steps 3
#  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4").
source "$(dirname "$0")/../../common.sh"
exec python -m sglang.launch_server --model-path "$VINFER_MODEL" "${REVISION_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --mamba-radix-cache-strategy extra_buffer \
  --page-size 64 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --random-seed 0 \
  "${HARNESS_ARGS[@]}" \
  --host 127.0.0.1 --port "$VINFER_PORT"

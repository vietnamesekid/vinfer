#!/usr/bin/env bash
# SGLang "default": cookbook command for Qwen3.5 (single GPU, no tp), unchanged except host/port.
# Source: sglang docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx at v0.5.23.
# Note (v0.5.23 source, arg_groups/overrides.py): radix cache is on, and for hybrid models
# --mamba-radix-cache-strategy auto picks extra_buffer when supported, else no_buffer.
source "$(dirname "$0")/../../common.sh"
exec python -m sglang.launch_server --model-path "$VINFER_MODEL" "${REVISION_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --random-seed 0 \
  "${HARNESS_ARGS[@]}" \
  --host 127.0.0.1 --port "$VINFER_PORT"

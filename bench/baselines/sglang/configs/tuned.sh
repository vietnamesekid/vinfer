#!/usr/bin/env bash
# SGLang "tuned": cookbook configuration tips for Qwen3.5 made explicit.
# Source: Qwen3.5.mdx §3.2 "Mamba Radix Cache" V2 (extra_buffer + page size 64, FLA backend,
# NVIDIA only) and the --mem-fraction-static tip.
# TODO: sweep --mem-fraction-static and --max-running-requests on the benchmark machine,
# record the sweep in docs/benchmarking.md.
source "$(dirname "$0")/../../common.sh"
exec python -m sglang.launch_server --model-path "$VINFER_MODEL" "${REVISION_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --mamba-radix-cache-strategy extra_buffer \
  --page-size 64 \
  --random-seed 0 \
  "${HARNESS_ARGS[@]}" \
  --host 127.0.0.1 --port "$VINFER_PORT"

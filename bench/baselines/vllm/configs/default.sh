#!/usr/bin/env bash
# vLLM "default": the command from the official recipe, unchanged except host/port/revision/seed,
# the Rust frontend and the harness settings (both in ../harness.sh).
# Source: https://recipes.vllm.ai/Qwen/Qwen3.5-2B (updated 2026-10-27, vLLM >= 0.17.0).
# Note (vLLM v0.33.0 source): prefix caching is on by default; for hybrid models it runs the
# Mamba state cache in "align" mode, which the recipe calls experimental.
source "$(dirname "$0")/../../common.sh"
exec vllm serve "$VINFER_MODEL" "${REVISION_ARGS[@]}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --seed 0 \
  "${HARNESS_ARGS[@]}" \
  --host 127.0.0.1 --port "$VINFER_PORT"

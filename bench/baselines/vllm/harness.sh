# What the benchmark harness needs from every vLLM config. Sourced by ../common.sh, so a
# config only carries its own tuning flags and appends "${HARNESS_ARGS[@]}".
#
# Dev mode exposes POST /reset_prefix_cache, which the harness calls before every measured run
# so no run starts with a warm cache. The server binds 127.0.0.1 only.
# (vllm/entrypoints/serve/__init__.py, v0.33.0)
export VLLM_SERVER_DEV_MODE=1
# Report usage.prompt_tokens_details.cached_tokens, for the prefix-cache hit rate.
# (vllm/entrypoints/common/cli_args.py)
HARNESS_ARGS=(--enable-prompt-tokens-details)
# Not set: VLLM_USE_RUST_FRONTEND=1 from the recipes (experimental vllm-rs frontend). It needs
# a binary we could not verify in the wheel, and the default Python frontend is what the
# `vllm bench serve` comparison (`make validate`) assumes.

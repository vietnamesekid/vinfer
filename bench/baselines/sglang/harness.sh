# What the benchmark harness needs from every SGLang config. Sourced by ../common.sh, so a
# config only carries its own tuning flags and appends "${HARNESS_ARGS[@]}".
#
# Report cached prompt tokens in usage, for the prefix-cache hit rate.
# (sglang/srt/arg_groups/fields/serving.py, v0.5.23)
# The cache reset endpoint (/flush_cache) needs no flag.
HARNESS_ARGS=(--enable-cache-report)

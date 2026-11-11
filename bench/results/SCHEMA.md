# Result file schema

`bench/results/<YYYY-MM-DDTHHMMSS.mmm>_<engine>-<config>_<org--model>_<workload>_c<concurrency>.json`
(the start time of the run; an existing file is never overwritten)

Written by `bench/harness/benchmark.py`, validated by `bench/harness/result_file.py:validate`. A run that
fails validation (dirty tree, < 3 runs, not dedicated, failed requests, missing model revision)
goes to `bench/results/raw/unofficial/` with its `problems` listed, never to `bench/results/`.

```json
{
  "engine": "vllm | sglang | vinfer | external | fake",
  "engine_version": "git sha or package version",
  "engine_config": {"config": "default | tuned | tuned-mtp",
                    "launch_script": "bench/baselines/<engine>/configs/<config>.sh",
                    "launch_script_text": "exact config script, every tuning flag",
                    "harness_script_text": "bench/baselines/<engine>/harness.sh: flags the harness needs",
                    "launch_script_sha256": "sha256 of both scripts",
                    "extra_body": null, "cache_reset_between_runs": "/reset_prefix_cache | null"},
  "vinfer_commit": "full git sha (dirty tree not allowed)",
  "dirty": false,
  "model": "<hf-model-id>",
  "model_revision": "HF commit sha",
  "workload": "W1-v1",
  "concurrency": 32,
  "arrival": "closed-loop | poisson",
  "request_rate": "req/s for poisson, null for closed-loop",
  "seed": 0,
  "runs": 3,
  "num_requests": 256,
  "num_requests_rule": "auto | <int> | null (auto = clamp(8 x concurrency, 64, 512))",
  "warmup": 4,
  "command": "exact command line",
  "hardware": {
    "provider": "...", "instance_type": "...", "dedicated": true,
    "gpu": "...", "gpu_memory": "...", "num_gpus": 1, "driver": "...", "cuda": "...",
    "pod_id": "..."
  },
  "software": {"python": "...", "torch": "..."},
  "slo": {"ttft_ms": null, "tpot_ms": null},
  "metrics": {
    "ttft_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
    "tpot_ms": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
    "output_tokens_per_s": 0.0,
    "goodput_req_per_s": 0.0,
    "peak_vram_gb": 0.0,
    "prefix_cache_hit_rate": null,
    "itl_ms": {"mean": 0.0, "p50": 0.0, "p99": 0.0},
    "e2el_ms": {"p50": 0.0, "p99": 0.0},
    "queue_ms": {"p50": 0.0, "p99": 0.0},
    "input_tokens_per_s": 0.0, "request_per_s": 0.0,
    "completed": 0, "failed": 0, "duration_s": 0.0
  },
  "stddev": {"...": "same keys as metrics: sample stddev across runs"},
  "per_run": [{"...": "metrics of each run"}],
  "raw_dir": "bench/results/raw/... (per-request records, not committed)",
  "problems": [],
  "timestamp": "ISO 8601"
}
```

Metric definitions:

Timing follows vLLM's `vllm bench serve` (v0.33.0) so the two can be compared (`make validate`):

- **TTFT**: request sent to the server → first output chunk. Time a request spends waiting for
  the client-side concurrency cap (or, in W4, for the previous turn) is `queue_ms`, not TTFT.
- **TPOT**: per request, (latency − TTFT) / (output tokens − 1); `tpot_ms.mean` averages requests.
- **Throughput**: output tokens/s for the whole system at the given concurrency.
- **Goodput**: requests/s meeting both the TTFT and TPOT SLOs.
- **Peak VRAM**: highest GPU memory during the run.
- **Prefix cache hit rate**: share of prompt tokens served from the prefix cache, from
  `usage.prompt_tokens_details.cached_tokens`; null if the server does not report it.
- **Peak VRAM**: NVML, all processes on the GPU. For vLLM/SGLang this mostly reflects memory
  reserved up front, read it with the engine config.

"""Benchmark harness: measure any OpenAI-compatible server on fixed workloads.

Entry point: ``python -m bench.harness {run,validate,fake-server}`` (see ``__main__.py``), or the
``make bench`` / ``make validate`` targets.

Modules, from the bottom up:

- ``config``: paths, ``HarnessError``, the typed configs every command is built from.
- ``workload``: the ``Request`` type; loading and selecting requests of a workload.
- ``http_client``: send one streaming chat request and time every output chunk.
- ``load_generator``: send many requests (Poisson arrivals, concurrency cap, session order).
- ``metrics``: TTFT / TPOT / throughput / goodput of one run, defined like ``vllm bench serve``.
- ``gpu_memory``: peak GPU memory during a run (NVML).
- ``environment``: git commit, hardware and software of the machine; ``.env.bench`` loading.
- ``engines``: the engines we can measure, and starting / stopping their servers.
- ``fake_server``: an in-process server with fixed delays, for CPU tests.
- ``result_file``: aggregate runs into one result, validate it, write it.
- ``benchmark``: the ``run`` command (concurrency sweep x runs).
- ``engine_benchmarks``: other engines' own benchmark tools, wrapped for comparison.
- ``validation``: the ``validate`` command (our harness against engines' own tools).
"""

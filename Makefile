# Rented machines: scripts/setup.sh writes KEY=VALUE settings here (bench/HARDWARE.md).
-include .env.bench
export

.PHONY: install lint fmt test parity noise workloads validate bench bench-all

MODEL ?=
REVISION ?=
KERNELS ?=
ENGINE ?= vllm
CONFIG ?= default
WORKLOAD ?= W1-v1
CONCURRENCY ?= 1 8 32 128
RUNS ?= 3
NUM_REQUESTS ?= auto
BENCH_ARGS ?=
WORKLOADS ?= W1-v1 W2-v1 W3-v1 W4-v1

# Every extra: dev tools, harness (bench), parity (transformers), workload generation.
# A plain `uv sync --extra dev` would remove what the tests and the harness import.
install:
	uv sync --locked --extra dev --extra bench --extra parity --extra workloads --extra server

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest tests/unit tests/regression -m "not gpu"

parity:
	@test -n "$(MODEL)" || (echo "usage: make parity MODEL=<hf-model-id>" && exit 1)
	VINFER_PARITY_MODEL=$(MODEL) VINFER_PARITY_REVISION=$(REVISION) VINFER_PARITY_KERNELS=$(KERNELS) \
		uv run pytest -m gpu -s tests/parity

# HF logit noise floor (GPU) -> bench/results/raw/noise-floor-<fla|torch>.json
noise:
	@test -n "$(MODEL)" || (echo "usage: make noise MODEL=<hf-model-id> [REVISION=sha]" && exit 1)
	uv run python -m tests.parity.noise_floor --model $(MODEL) $(if $(REVISION),--revision $(REVISION))

# Build the workloads from pinned sources; CHECK=1 only verifies them against the manifests.
workloads:
	uv run python -m bench.workloads.generate $(if $(CHECK),--check) $(WORKLOADS)

# Compare our harness with each engine's own benchmark tool (GPU).
BENCHMARKS ?= vllm sglang
validate:
	@test -n "$(MODEL)" || (echo "usage: make validate MODEL=<hf-id> [BENCHMARKS='vllm sglang']" && exit 1)
	uv run python -m bench.harness validate --model $(MODEL) --benchmarks $(BENCHMARKS)

# One engine/config on one or more workloads (WORKLOAD="W1-v1 W4-v1"). See bench/README.md.
bench:
	@test -n "$(MODEL)" || (echo "usage: make bench MODEL=<hf-id> ENGINE=vllm CONFIG=tuned WORKLOAD=W1-v1" && exit 1)
	uv run python -m bench.harness run --engine $(ENGINE) --config $(CONFIG) --model $(MODEL) \
		--workload $(WORKLOAD) --concurrency $(CONCURRENCY) --runs $(RUNS) \
		--num-requests $(NUM_REQUESTS) $(BENCH_ARGS)

# Full baseline matrix, in one rental session: see scripts/bench_matrix.sh for the
# ENGINES / CONFIGS / WORKLOADS overrides. Keeps going past failures.
bench-all:
	@test -n "$(MODEL)" || (echo "usage: make bench-all MODEL=<hf-id>" && exit 1)
	MODEL=$(MODEL) scripts/bench_matrix.sh

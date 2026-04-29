.PHONY: test-fast test-integration test-full lock-deps run-fast bench-latency

PYTHON ?= python
ifneq ("$(wildcard .venv/bin/python)","")
PYTHON := .venv/bin/python
endif

API_BASE_URL ?= http://127.0.0.1:8000/api/v1
BENCH_MODE ?= compare
BENCH_RUNS ?= 5
BENCH_WARMUP ?= 1
BENCH_CONCURRENCY ?= 1
BENCH_STREAM ?= false
BENCH_OUTPUT_JSON ?=

test-fast:
	$(PYTHON) -m pytest -m "not slow and not e2e"

test-integration:
	$(PYTHON) -m pytest -m "integration and not slow and not e2e"

test-full:
	$(PYTHON) -m pytest

lock-deps:
	$(PYTHON) scripts/lock_requirements.py

run-fast:
	@if [ ! -f .env ]; then \
		cp .env.fast.example .env; \
		echo "Created .env from .env.fast.example"; \
	fi
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

bench-latency:
	@STREAM_FLAG=""; \
	if [ "$(BENCH_STREAM)" = "true" ]; then STREAM_FLAG="--stream"; fi; \
	OUTPUT_FLAG=""; \
	if [ -n "$(BENCH_OUTPUT_JSON)" ]; then OUTPUT_FLAG="--output-json $(BENCH_OUTPUT_JSON)"; fi; \
	$(PYTHON) scripts/benchmark_latency.py \
		--api-base-url "$(API_BASE_URL)" \
		--mode "$(BENCH_MODE)" \
		--runs "$(BENCH_RUNS)" \
		--warmup "$(BENCH_WARMUP)" \
		--concurrency "$(BENCH_CONCURRENCY)" \
		$$STREAM_FLAG \
		$$OUTPUT_FLAG

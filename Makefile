.PHONY: test-fast test-integration test-full

PYTHON ?= python
ifneq ("$(wildcard .venv/bin/python)","")
PYTHON := .venv/bin/python
endif

test-fast:
	$(PYTHON) -m pytest -m "not slow and not e2e"

test-integration:
	$(PYTHON) -m pytest -m "integration and not slow and not e2e"

test-full:
	$(PYTHON) -m pytest

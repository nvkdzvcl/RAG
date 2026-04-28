"""Shared pytest configuration for test categorization and fast local defaults."""

from __future__ import annotations

import os
from collections.abc import Generator, Iterable

import pytest

from app.core.config import get_settings

INTEGRATION_PATH_PREFIXES: tuple[str, ...] = (
    "tests/api/",
    "tests/workflows/",
    "tests/services/",
)

E2E_PATHS: set[str] = {
    "tests/evaluation/test_runner.py",
}

# DOCX/PDF parser tests are intentionally marked slow for day-to-day loops.
SLOW_NODEIDS: set[str] = {
    "tests/ingestion/test_loaders.py::test_docx_loader_supports_vietnamese_text",
    "tests/ingestion/test_parsers.py::test_docx_parser_extracts_heading_paragraph_and_table",
    "tests/ingestion/test_directory_ingestor.py::test_directory_ingestor_loads_markdown_and_text",
    "tests/api/test_documents_routes.py::test_upload_endpoint_accepts_docx",
}

TEST_ENV_DEFAULTS: dict[str, str] = {
    "EMBEDDING_PROVIDER": "hash",
    "EMBEDDING_HASH_DIMENSION": "64",
    "RERANKER_ENABLED": "true",
    "RERANKER_PROVIDER": "score_only",
    "RERANKER_TOP_K": "6",
    "RERANKER_TOP_N": "6",
    "OCR_ENABLED": "false",
    "OCR_LANGUAGE": "vie+eng",
    "OCR_MIN_TEXT_CHARS": "100",
    "OCR_RENDER_DPI": "216",
    "TESSERACT_CMD": "",
    "OCR_CONFIDENCE_THRESHOLD": "40",
    "LLM_PROVIDER": "stub",
    "GROUNDING_SEMANTIC_ENABLED": "false",
}


def _apply_test_env_defaults() -> None:
    for key, value in TEST_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


# Apply once at import time so module-level TestClient(...) initializations
# also pick up fast defaults before loading app settings.
_apply_test_env_defaults()


def _normalized_path(nodeid: str) -> str:
    return nodeid.split("::", 1)[0].replace("\\", "/")


def _starts_with_any(value: str, prefixes: Iterable[str]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


@pytest.fixture(autouse=True)
def _fast_runtime_defaults() -> Generator[None, None, None]:
    """Keep local tests deterministic and avoid loading heavy model backends by default."""
    _apply_test_env_defaults()
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def pytest_configure() -> None:
    _apply_test_env_defaults()
    get_settings.cache_clear()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = _normalized_path(item.nodeid)

        if path in E2E_PATHS:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
            continue

        if _starts_with_any(path, INTEGRATION_PATH_PREFIXES):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        if item.nodeid in SLOW_NODEIDS:
            item.add_marker(pytest.mark.slow)

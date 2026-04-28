"""Tests for query service request mapping."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.schemas.api import QueryRequest, StandardQueryResponse
from app.schemas.common import Mode
from app.services.query_service import QueryService


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        query: str,
        mode: Mode,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
    ) -> StandardQueryResponse:
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "chat_history": chat_history,
                "model": model,
                "response_language": response_language,
                "query_filters": query_filters,
            }
        )
        return StandardQueryResponse(answer="ok")


def test_run_request_passes_model_and_non_empty_filters_to_runner() -> None:
    runner = _RecordingRunner()
    service = QueryService(runner=runner)  # type: ignore[arg-type]
    uploaded_after = datetime(2026, 1, 1, tzinfo=timezone.utc)
    uploaded_before = datetime(2026, 1, 31, tzinfo=timezone.utc)
    payload = QueryRequest(
        query="abc",
        mode=Mode.STANDARD,
        chat_history=[],
        model="qwen2.5:7b",
        doc_ids=["doc-a"],
        filenames=[],
        file_types=["pdf"],
        uploaded_after=uploaded_after,
        uploaded_before=uploaded_before,
        include_ocr=False,
    )

    response = service.run_request(payload)

    assert response.mode == "standard"
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["model"] == "qwen2.5:7b"
    assert call["query_filters"] == {
        "doc_ids": ["doc-a"],
        "file_types": ["pdf"],
        "uploaded_after": uploaded_after,
        "uploaded_before": uploaded_before,
        "include_ocr": False,
    }


def test_run_request_uses_none_for_empty_filters() -> None:
    runner = _RecordingRunner()
    service = QueryService(runner=runner)  # type: ignore[arg-type]
    payload = QueryRequest(query="abc", mode=Mode.STANDARD, chat_history=[])

    service.run_request(payload)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["query_filters"] is None

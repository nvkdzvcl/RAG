"""API tests for SSE query streaming route."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.api import StandardQueryResponse, validate_query_response


client = TestClient(create_app())


def _parse_sse_events(raw_stream: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    for line in raw_stream.splitlines():
        if not line:
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                events.append((event_name or "message", payload))
            event_name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line.split(":", maxsplit=1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", maxsplit=1)[1].strip())

    return events


def test_query_stream_returns_incremental_events_and_final_payload() -> None:
    payload = {
        "query": "What does the standard pipeline do?",
        "mode": "standard",
        "chat_history": [],
    }

    with client.stream("POST", "/api/v1/query/stream", json=payload) as response:
        assert response.status_code == 200
        stream_text = "".join(list(response.iter_text()))

    events = _parse_sse_events(stream_text)
    event_types = [name for name, _ in events]

    assert "start" in event_types
    assert "retrieval" in event_types
    assert "generation" in event_types
    assert "final" in event_types
    assert "done" in event_types

    final_payload = next(data["response"] for name, data in events if name == "final")
    parsed = validate_query_response(final_payload)

    assert isinstance(parsed, StandardQueryResponse)
    assert parsed.mode == "standard"
    assert parsed.answer


def test_query_stream_final_output_matches_regular_query_contract() -> None:
    payload = {
        "query": "What does the standard pipeline do?",
        "mode": "standard",
        "chat_history": [],
    }

    normal_response = client.post("/api/v1/query", json=payload)
    assert normal_response.status_code == 200
    normal_parsed = validate_query_response(normal_response.json())
    assert isinstance(normal_parsed, StandardQueryResponse)

    with client.stream("POST", "/api/v1/query/stream", json=payload) as stream_response:
        assert stream_response.status_code == 200
        stream_text = "".join(list(stream_response.iter_text()))

    events = _parse_sse_events(stream_text)
    final_payload = next(data["response"] for name, data in events if name == "final")
    streamed_parsed = validate_query_response(final_payload)
    assert isinstance(streamed_parsed, StandardQueryResponse)

    assert streamed_parsed.mode == normal_parsed.mode
    assert streamed_parsed.answer == normal_parsed.answer
    assert streamed_parsed.status == normal_parsed.status

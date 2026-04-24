"""API tests for standard-mode query route."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.api import StandardQueryResponse, validate_query_response


client = TestClient(create_app())


def test_query_route_standard_returns_contract_payload() -> None:
    payload = {
        "query": "What does the standard pipeline do?",
        "mode": "standard",
        "chat_history": [],
    }

    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 200

    body = response.json()
    parsed = validate_query_response(body)

    assert isinstance(parsed, StandardQueryResponse)
    assert parsed.mode == "standard"
    assert parsed.answer

"""API tests for advanced-mode query route."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.api import AdvancedQueryResponse, validate_query_response


client = TestClient(create_app())


def test_query_route_advanced_returns_contract_payload() -> None:
    payload = {
        "query": "How does advanced mode improve answer reliability?",
        "mode": "advanced",
        "chat_history": [],
    }

    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 200

    body = response.json()
    parsed = validate_query_response(body)

    assert isinstance(parsed, AdvancedQueryResponse)
    assert parsed.mode == "advanced"
    assert parsed.answer

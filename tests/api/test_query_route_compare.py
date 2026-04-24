"""API tests for compare-mode query route."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.api import CompareQueryResponse, validate_query_response


client = TestClient(create_app())


def test_query_route_compare_returns_both_branches() -> None:
    payload = {
        "query": "Compare the two workflows for this question",
        "mode": "compare",
        "chat_history": [],
    }

    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 200

    body = response.json()
    parsed = validate_query_response(body)

    assert isinstance(parsed, CompareQueryResponse)
    assert parsed.mode == "compare"
    assert parsed.standard.mode == "standard"
    assert parsed.advanced.mode == "advanced"

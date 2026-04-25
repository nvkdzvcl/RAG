"""API tests for health route."""

from fastapi.testclient import TestClient

from app.main import create_app


client = TestClient(create_app())


def test_health_route_returns_ok() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body.get("llm_model"), str)
    assert body["llm_model"]

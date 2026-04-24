"""Tests for standard workflow end-to-end path."""

from app.schemas.api import StandardQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.services import QueryService
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


def test_standard_workflow_run_path() -> None:
    workflow = StandardWorkflow()

    response = workflow.run(query="How does standard mode perform retrieval?", chat_history=None)

    assert isinstance(response, StandardQueryResponse)
    assert response.mode == "standard"
    assert response.answer
    assert isinstance(response.citations, list)
    assert response.status in {"answered", "partial", "insufficient_evidence"}


def test_standard_runner_route_and_contract() -> None:
    runner = WorkflowRunner()

    response = runner.run(query="What is hybrid retrieval?", mode=Mode.STANDARD)
    parsed = validate_query_response(response.model_dump())

    assert isinstance(parsed, StandardQueryResponse)
    assert parsed.mode == "standard"


def test_query_service_standard_mode() -> None:
    service = QueryService()

    response = service.run(query="Explain citation grounding", mode=Mode.STANDARD)

    assert response.mode == "standard"
    assert response.answer

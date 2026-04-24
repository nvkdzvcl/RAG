"""Compare workflow tests."""

from app.schemas.api import CompareQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.workflows.runner import WorkflowRunner


def test_compare_mode_response_contains_both_branches() -> None:
    runner = WorkflowRunner()

    response = runner.run(query="Compare standard and advanced reliability", mode=Mode.COMPARE)

    assert isinstance(response, CompareQueryResponse)
    assert response.mode == "compare"
    assert response.standard.mode == "standard"
    assert response.advanced.mode == "advanced"
    assert response.standard.answer
    assert response.advanced.answer


def test_compare_mode_schema_contract() -> None:
    runner = WorkflowRunner()
    response = runner.run(query="What is compare mode?", mode=Mode.COMPARE)

    parsed = validate_query_response(response.model_dump())
    assert isinstance(parsed, CompareQueryResponse)
    assert parsed.standard.mode == "standard"
    assert parsed.advanced.mode == "advanced"
    assert parsed.comparison.citation_delta == (
        len(parsed.advanced.citations) - len(parsed.standard.citations)
    )

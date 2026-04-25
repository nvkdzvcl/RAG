"""Compare workflow tests."""

from app.schemas.api import CompareQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult
from app.retrieval import ScoreOnlyReranker
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.compare import CompareWorkflow
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


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


def test_compare_workflow_uses_injected_qwen_backed_branches() -> None:
    class _MockQwenClient:
        def complete(self, prompt: str, system_prompt: str | None = None) -> str:
            _ = prompt
            _ = system_prompt
            return '{"answer":"Qwen DI answer.","confidence":0.81,"status":"answered"}'

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="c_cmp_001",
                    doc_id="d_cmp_001",
                    source="seeded://cmp",
                    content="Compare mode uses standard and advanced branches.",
                    score=0.9,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_MockQwenClient(),
        reranker=ScoreOnlyReranker(),
    )
    advanced = AdvancedWorkflow(standard_workflow=standard, max_loops=1)
    compare = CompareWorkflow(
        standard_workflow=standard,
        advanced_workflow=advanced,
    )

    response = compare.run(query="compare via di", chat_history=None)

    assert response.standard.answer == "Qwen DI answer."
    assert response.advanced.answer == "Qwen DI answer."

"""Advanced workflow tests."""

from app.schemas.api import AdvancedQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import CritiqueResult
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.critique import HeuristicCritic
from app.workflows.runner import WorkflowRunner


def _sample_context() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk_adv_001",
            doc_id="doc_adv_001",
            source="memory://adv1",
            title="Advanced Evidence",
            section="Main",
            content="Advanced workflow retries retrieval and critiques groundedness.",
            score=0.7,
            score_type="hybrid",
        )
    ]


def test_critique_schema_validity() -> None:
    critic = HeuristicCritic()

    critique = critic.critique(
        query="How does advanced retrieval work?",
        draft_answer="Advanced workflow retries retrieval.",
        context=_sample_context(),
        loop_count=1,
        max_loops=2,
    )

    assert isinstance(critique, CritiqueResult)
    dumped = critique.model_dump()
    for key in [
        "grounded",
        "enough_evidence",
        "has_conflict",
        "missing_aspects",
        "should_retry_retrieval",
        "should_refine_answer",
        "better_queries",
        "confidence",
        "note",
    ]:
        assert key in dumped


def test_retry_loop_limit() -> None:
    workflow = AdvancedWorkflow(max_loops=2)

    response = workflow.run("force retry retrieval loop for diagnostics")

    assert isinstance(response, AdvancedQueryResponse)
    assert response.loop_count == 2
    assert response.stop_reason == "max_loop_reached"


def test_abstain_behavior() -> None:
    workflow = AdvancedWorkflow(max_loops=2)

    response = workflow.run("force abstain on unsupported claim")

    assert response.status == "insufficient_evidence"
    assert response.citations == []
    assert response.stop_reason == "critic_abstain"


def test_advanced_mode_run_path() -> None:
    runner = WorkflowRunner()

    response = runner.run(query="How does advanced mode improve reliability?", mode=Mode.ADVANCED)
    parsed = validate_query_response(response.model_dump())

    assert isinstance(parsed, AdvancedQueryResponse)
    assert parsed.mode == "advanced"
    assert parsed.loop_count <= 2
    assert parsed.answer

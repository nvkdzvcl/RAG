"""Workflow tests for LLM fallback behavior and shared client wiring."""

from __future__ import annotations

from app.generation import FallbackLLMClient, StubLLMClient
from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


class _FailingPrimaryLLM:
    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        _ = prompt
        _ = system_prompt
        raise RuntimeError("endpoint unavailable")


def _build_safe_fallback_client() -> FallbackLLMClient:
    fallback = StubLLMClient(
        responder=lambda prompt, system: (
            '{"answer":"Fallback answer","confidence":0.4,"status":"answered"}'
        )
    )
    return FallbackLLMClient(primary=_FailingPrimaryLLM(), fallback=fallback)


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        _ = query
        _ = top_k
        return [
            RetrievalResult(
                chunk_id="c-llm-001",
                doc_id="d-llm-001",
                source="seeded://llm",
                content="This context supports mocked Qwen answers.",
                score=0.92,
                score_type="hybrid",
                rank=1,
            )
        ]


class _FakeIndexManager:
    def get_retriever(self) -> _FakeRetriever:
        return _FakeRetriever()

    def get_active_source(self) -> str:
        return "seeded"


def test_workflow_runner_runs_all_modes_without_real_qwen(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.workflows.standard.create_llm_client_from_settings",
        lambda settings: _build_safe_fallback_client(),
    )

    runner = WorkflowRunner()

    standard = runner.run("Standard mode question", Mode.STANDARD)
    advanced = runner.run("Advanced mode question", Mode.ADVANCED)
    compare = runner.run("Compare modes question", Mode.COMPARE)

    assert standard.answer
    assert advanced.answer
    assert compare.standard.answer
    assert compare.advanced.answer


def test_standard_workflow_falls_back_when_qwen_endpoint_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.workflows.standard.create_llm_client_from_settings",
        lambda settings: _build_safe_fallback_client(),
    )

    workflow = StandardWorkflow(index_manager=_FakeIndexManager())
    response = workflow.run("fallback test")

    assert "Fallback answer" in response.answer


def test_compare_mode_uses_mocked_qwen_for_both_branches(monkeypatch) -> None:
    mocked_qwen_client = StubLLMClient(
        responder=lambda prompt, system: (
            '{"answer":"Mocked Qwen answer.","confidence":0.77,"status":"answered"}'
        )
    )
    monkeypatch.setattr(
        "app.workflows.standard.create_llm_client_from_settings",
        lambda settings: mocked_qwen_client,
    )

    runner = WorkflowRunner(index_manager=_FakeIndexManager())
    response = runner.run("compare with mocked qwen", Mode.COMPARE)

    assert "Mocked Qwen answer." in response.standard.answer
    assert "Mocked Qwen answer." in response.advanced.answer


def test_advanced_components_share_standard_llm_client_instance() -> None:
    llm_client = StubLLMClient()
    standard = StandardWorkflow(llm_client=llm_client)
    advanced = AdvancedWorkflow(standard_workflow=standard)

    assert advanced.retrieval_gate.llm_client is llm_client
    assert advanced.query_rewriter.llm_client is llm_client
    assert advanced.critic.llm_client is llm_client
    assert advanced.refiner.llm_client is llm_client

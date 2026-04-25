"""Workflow tests for LLM fallback behavior and shared client wiring."""

from __future__ import annotations

from app.generation import FallbackLLMClient, StubLLMClient
from app.schemas.common import Mode
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
        responder=lambda prompt, system: '{"answer":"Fallback answer","confidence":0.4,"status":"answered"}'
    )
    return FallbackLLMClient(primary=_FailingPrimaryLLM(), fallback=fallback)


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


def test_advanced_components_share_standard_llm_client_instance() -> None:
    llm_client = StubLLMClient()
    standard = StandardWorkflow(llm_client=llm_client)
    advanced = AdvancedWorkflow(standard_workflow=standard)

    assert advanced.retrieval_gate.llm_client is llm_client
    assert advanced.query_rewriter.llm_client is llm_client
    assert advanced.critic.llm_client is llm_client
    assert advanced.refiner.llm_client is llm_client

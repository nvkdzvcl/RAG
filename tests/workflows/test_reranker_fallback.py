"""Workflow-level tests for reranker fallback behavior."""

from __future__ import annotations

from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult
from app.retrieval import ScoreOnlyReranker
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


def test_workflows_run_when_cross_encoder_unavailable(monkeypatch) -> None:
    class _BrokenCrossEncoderReranker:
        def __init__(self, **_: object) -> None:
            raise RuntimeError("cross encoder unavailable")

    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoderReranker",
        _BrokenCrossEncoderReranker,
    )

    runner = WorkflowRunner()

    standard = runner.run(query="What is standard mode?", mode=Mode.STANDARD)
    advanced = runner.run(query="What is advanced mode?", mode=Mode.ADVANCED)
    compare = runner.run(query="Compare these modes", mode=Mode.COMPARE)

    assert standard.answer
    assert advanced.answer
    assert compare.standard.answer
    assert compare.advanced.answer


def test_standard_workflow_uses_create_reranker_from_settings(monkeypatch) -> None:
    class _Settings:
        corpus_dir = "docs"
        index_dir = "data/indexes"
        reranker_provider = "score_only"
        reranker_model = "stub-model"
        reranker_device = "cpu"
        reranker_batch_size = 3
        reranker_top_n = 2

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return []

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    captured: dict[str, object] = {}

    def _fake_create_reranker(*, provider_name: str, model: str, device: str, batch_size: int):
        captured.update(
            {
                "provider_name": provider_name,
                "model": model,
                "device": device,
                "batch_size": batch_size,
            }
        )
        return ScoreOnlyReranker()

    monkeypatch.setattr("app.workflows.standard.get_settings", lambda: _Settings())
    monkeypatch.setattr("app.workflows.standard.create_reranker", _fake_create_reranker)

    workflow = StandardWorkflow(index_manager=_FakeIndexManager())
    response = workflow.run(query="test", chat_history=None)

    assert response.mode == "standard"
    assert captured == {
        "provider_name": "score_only",
        "model": "stub-model",
        "device": "cpu",
        "batch_size": 3,
    }


def test_advanced_workflow_reuses_standard_workflow_configured_reranker() -> None:
    class _SpyReranker(ScoreOnlyReranker):
        def __init__(self) -> None:
            super().__init__()
            self.called = 0

        def rerank(self, query: str, docs: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
            self.called += 1
            return super().rerank(query, docs, top_k=top_k)

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="c1",
                    doc_id="d1",
                    source="seeded://doc",
                    content="Advanced mode can retry retrieval before finalizing answers.",
                    score=0.8,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    spy_reranker = _SpyReranker()
    standard = StandardWorkflow(index_manager=_FakeIndexManager(), reranker=spy_reranker)
    advanced = AdvancedWorkflow(standard_workflow=standard, max_loops=1)

    response = advanced.run("How does advanced mode improve reliability?")

    assert response.mode == "advanced"
    assert spy_reranker.called >= 1

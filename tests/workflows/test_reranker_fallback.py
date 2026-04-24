"""Workflow-level tests for reranker fallback behavior."""

from __future__ import annotations

from app.schemas.common import Mode
from app.workflows.runner import WorkflowRunner


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

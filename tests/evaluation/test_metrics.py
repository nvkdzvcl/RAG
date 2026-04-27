"""Unit tests for evaluation metric computation."""

from app.evaluation.metrics import compute_metrics, extract_trace_fields
from app.schemas.common import Citation


def test_extract_trace_fields_and_metrics() -> None:
    trace = [
        {
            "step": "retrieve",
            "count": 3,
            "chunk_ids": ["c1", "c2", "c3"],
        },
        {
            "step": "rerank",
            "docs": [
                {"chunk_id": "c2", "rerank_score": 0.91},
                {"chunk_id": "c1", "rerank_score": 0.41},
            ],
        },
        {
            "step": "context_select",
            "count": 2,
            "docs": [
                {
                    "chunk_id": "c2",
                    "content": "Self-RAG supports retrieval and critique loops.",
                },
                {
                    "chunk_id": "c1",
                    "content": "Advanced mode can retry when evidence is weak.",
                },
            ],
        },
    ]
    fields = extract_trace_fields(trace)

    assert fields.retrieved_count == 3
    assert fields.selected_context_count == 2
    assert fields.rerank_scores["c2"] == 0.91

    metrics = compute_metrics(
        expected_behavior="answer",
        answer="Advanced mode can retry retrieval.",
        citations=[
            Citation(
                chunk_id="c2",
                doc_id="doc_01",
                source="docs/MODES.md",
            )
        ],
        confidence=0.8,
        status="answered",
        loop_count=2,
        stop_reason="critique_pass",
        latency_ms=180,
        trace_fields=fields,
        reference_answer="Advanced mode retries retrieval with critique.",
        gold_sources=["docs/MODES.md"],
    )

    assert metrics.citation_count == 1
    assert metrics.has_citations is True
    assert metrics.retry_used is True
    assert metrics.answer_non_empty is True
    assert metrics.answer_contains_reference_keywords is True
    assert metrics.cited_gold_source_overlap == 1.0
    assert metrics.groundedness_proxy is not None

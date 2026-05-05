"""Tests for context selection truncation safety rules."""

from app.retrieval.context_selector import ContextSelector
from app.schemas.retrieval import RetrievalResult


def _result(
    *,
    chunk_id: str,
    content: str,
    block_type: str = "text",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        source=f"memory://{chunk_id}",
        content=content,
        metadata={"block_type": block_type},
        score=1.0,
        score_type="hybrid",
    )


def test_context_selector_skips_oversized_table_instead_of_truncating() -> None:
    table = _result(
        chunk_id="table_1",
        content=(
            "| Col A | Col B |\n| --- | --- |\n"
            + "\n".join(f"| value{i} | detail{i} |" for i in range(30))
        ),
        block_type="table",
    )
    text = _result(
        chunk_id="text_1",
        content="This supporting paragraph should still be selected.",
        block_type="text",
    )

    selected = ContextSelector(
        max_chunks=3,
        max_chars=120,
        min_useful_chars=20,
    ).select([table, text])

    assert selected
    assert all(item.chunk_id != "table_1" for item in selected)
    assert any(item.chunk_id == "text_1" for item in selected)
    assert all(not bool(item.metadata.get("truncated")) for item in selected)


def test_context_selector_can_still_truncate_text_chunks_when_needed() -> None:
    text = _result(
        chunk_id="text_long",
        content=(
            "Sentence one with useful details. "
            "Sentence two extends beyond the configured context budget."
        ),
        block_type="text",
    )

    selected = ContextSelector(
        max_chunks=1,
        max_chars=48,
        min_useful_chars=20,
    ).select([text])

    assert len(selected) == 1
    assert selected[0].chunk_id == "text_long"
    assert bool(selected[0].metadata.get("truncated")) is True
    assert len(selected[0].content) <= 48

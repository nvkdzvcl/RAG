"""BM25 tokenization tests."""

from __future__ import annotations

import app.indexing.bm25_index as bm25_module
from app.indexing.bm25_index import BM25Index, tokenize_bm25
from app.retrieval.sparse import SparseRetriever
from app.schemas.ingestion import DocumentChunk


def _chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        source=f"memory://{chunk_id}",
        content=content,
    )


def test_tokenize_bm25_uses_underthesea_when_available(monkeypatch) -> None:
    mapping = {
        "sinh viên đại học": "sinh_viên đại_học",
    }

    def fake_word_tokenize(text: str, format: str = "text") -> str:
        _ = format
        return mapping.get(text, text)

    monkeypatch.setattr(bm25_module, "_UNDERTHESEA_WORD_TOKENIZE", fake_word_tokenize)

    assert tokenize_bm25("sinh viên đại học") == ["sinh_viên", "đại_học"]
    assert BM25Index()._tokenize("sinh viên đại học") == ["sinh_viên", "đại_học"]


def test_tokenize_bm25_falls_back_when_underthesea_missing(monkeypatch) -> None:
    monkeypatch.setattr(bm25_module, "_UNDERTHESEA_WORD_TOKENIZE", None)

    assert tokenize_bm25("sinh viên đại học") == ["sinh", "viên", "đại", "học"]


def test_sparse_retriever_uses_segmented_tokens_for_vietnamese(monkeypatch) -> None:
    mapping = {
        "sinh viên đại học": "sinh_viên đại_học",
        "sinh viên trung học": "sinh_viên trung_học",
    }

    def fake_word_tokenize(text: str, format: str = "text") -> str:
        _ = format
        return mapping.get(text, text)

    monkeypatch.setattr(bm25_module, "_UNDERTHESEA_WORD_TOKENIZE", fake_word_tokenize)

    index = BM25Index()
    index.build(
        [
            _chunk("chunk_vi_1", "sinh viên đại học"),
            _chunk("chunk_vi_2", "sinh viên trung học"),
        ]
    )
    results = SparseRetriever(index).retrieve("sinh viên đại học", top_k=2)

    assert [item.chunk_id for item in results] == ["chunk_vi_1", "chunk_vi_2"]

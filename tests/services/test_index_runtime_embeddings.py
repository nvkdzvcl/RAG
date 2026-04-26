"""Tests for runtime index embedding provider integration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.core.config import get_settings
from app.ingestion.base_loader import build_doc_id
from app.indexing import BaseEmbeddingProvider
from app.services.index_runtime import RuntimeIndexManager


class _CountingEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 6) -> None:
        self.name = "counting-provider"
        self.dimension = dimension
        self.document_inputs: list[str] = []
        self.query_inputs: list[str] = []

    def _vectorize(self, text: str) -> list[float]:
        base = sum(ord(char) for char in text) % 997
        return [float((base + index) % 97) for index in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs.extend(texts)
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        return self._vectorize(text)


class _NoRebuildEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 6) -> None:
        self.name = "counting-provider"
        self.dimension = dimension
        self.query_inputs: list[str] = []

    def _vectorize(self, text: str) -> list[float]:
        base = sum(ord(char) for char in text) % 997
        return [float((base + index) % 97) for index in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        _ = texts
        raise AssertionError("embed_documents should not be called when loading matching persisted uploaded index")

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        return self._vectorize(text)


def test_runtime_index_build_uses_embedding_provider_interface(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "indexes"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "vi.txt").write_text(
        "Tài liệu tiếng Việt mô tả hệ thống truy hồi thông tin cho câu hỏi hỗn hợp Việt-Anh.",
        encoding="utf-8",
    )

    provider = _CountingEmbeddingProvider(dimension=6)
    manager = RuntimeIndexManager(
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        embedding_provider=provider,
    )

    chunk_count = manager.activate_from_seeded_corpus()
    results = manager.get_retriever().retrieve("Truy hoi thong tin la gi?", top_k=3)

    assert chunk_count > 0
    assert manager.get_active_source() == "seeded"
    assert manager.get_active_chunk_count() == chunk_count
    assert provider.document_inputs
    assert provider.query_inputs
    assert results
    assert (index_dir / "vector_index.json").exists()
    assert (index_dir / "bm25_index.json").exists()


def test_runtime_index_manager_uses_configured_provider_factory(monkeypatch, tmp_path: Path) -> None:
    class _Settings:
        embedding_provider = "hash"
        embedding_model = "intfloat/multilingual-e5-base"
        embedding_device = "cpu"
        embedding_batch_size = 32
        embedding_normalize = False
        embedding_hash_dimension = 79

    captured: dict[str, object] = {}
    configured_provider = _CountingEmbeddingProvider(dimension=79)

    def _fake_factory(
        *,
        provider_name: str,
        model: str,
        device: str,
        batch_size: int,
        normalize: bool,
        fallback_hash_dimension: int,
    ) -> BaseEmbeddingProvider:
        captured.update(
            {
                "provider_name": provider_name,
                "model": model,
                "device": device,
                "batch_size": batch_size,
                "normalize": normalize,
                "fallback_hash_dimension": fallback_hash_dimension,
            }
        )
        return configured_provider

    monkeypatch.setattr("app.services.index_runtime.get_settings", lambda: _Settings())
    monkeypatch.setattr("app.services.index_runtime.create_embedding_provider", _fake_factory)

    manager = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
    )

    assert manager.embedding_provider is configured_provider
    assert captured == {
        "provider_name": "hash",
        "model": "intfloat/multilingual-e5-base",
        "device": "cpu",
        "batch_size": 32,
        "normalize": False,
        "fallback_hash_dimension": 79,
    }


def test_runtime_index_manager_uses_sentence_transformers_when_configured(monkeypatch, tmp_path: Path) -> None:
    class _Settings:
        embedding_provider = "sentence_transformers"
        embedding_model = "intfloat/multilingual-e5-base"
        embedding_device = "cpu"
        embedding_batch_size = 16
        embedding_normalize = True
        embedding_hash_dimension = 64

    captured: dict[str, object] = {}
    configured_provider = _CountingEmbeddingProvider(dimension=12)

    def _fake_factory(
        *,
        provider_name: str,
        model: str,
        device: str,
        batch_size: int,
        normalize: bool,
        fallback_hash_dimension: int,
    ) -> BaseEmbeddingProvider:
        captured.update(
            {
                "provider_name": provider_name,
                "model": model,
                "device": device,
                "batch_size": batch_size,
                "normalize": normalize,
                "fallback_hash_dimension": fallback_hash_dimension,
            }
        )
        return configured_provider

    monkeypatch.setattr("app.services.index_runtime.get_settings", lambda: _Settings())
    monkeypatch.setattr("app.services.index_runtime.create_embedding_provider", _fake_factory)

    manager = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
    )

    assert manager.embedding_provider is configured_provider
    assert captured["provider_name"] == "sentence_transformers"
    assert captured["model"] == "intfloat/multilingual-e5-base"


def test_runtime_index_manager_indexes_and_retrieves_ocr_chunks(monkeypatch, tmp_path: Path) -> None:
    class _Settings:
        embedding_provider = "hash"
        embedding_model = "intfloat/multilingual-e5-base"
        embedding_device = "cpu"
        embedding_batch_size = 16
        embedding_normalize = True
        embedding_hash_dimension = 64
        ocr_enabled = True
        ocr_language = "vie+eng"
        ocr_min_text_chars = 100
        ocr_render_dpi = 216
        tesseract_cmd = ""
        ocr_confidence_threshold = 40.0

    class FakePage:
        images = []

        @staticmethod
        def extract_text() -> str:
            return ""

        @staticmethod
        def extract_tables() -> list[list[list[str]]]:
            return []

    class FakePDF:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def __enter__(self) -> "FakePDF":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = exc_type
            _ = exc
            _ = tb

    monkeypatch.setattr("app.services.index_runtime.get_settings", lambda: _Settings())
    monkeypatch.setenv("OCR_ENABLED", "true")
    monkeypatch.setenv("OCR_LANGUAGE", "vie+eng")
    monkeypatch.setenv("OCR_MIN_TEXT_CHARS", "100")
    monkeypatch.setenv("OCR_RENDER_DPI", "216")
    monkeypatch.setenv("OCR_CONFIDENCE_THRESHOLD", "40")
    monkeypatch.setenv("TESSERACT_CMD", "")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "app.ingestion.parsers.pdf_parser.pdfplumber",
        SimpleNamespace(open=lambda _: FakePDF()),
    )
    monkeypatch.setattr("app.ingestion.parsers.pdf_parser.is_tesseract_available", lambda: True)
    monkeypatch.setattr(
        "app.ingestion.parsers.pdf_parser.ocr_pdf_page_with_pymupdf",
        lambda *args, **kwargs: "Nội dung OCR thử nghiệm với token duy nhất: ocrtokenviet99",
    )

    provider = _CountingEmbeddingProvider(dimension=8)
    manager = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
        embedding_provider=provider,
    )

    uploaded_pdf = tmp_path / "testocr.pdf"
    uploaded_pdf.write_bytes(b"%PDF-1.4 fake scan payload")

    indexed = manager.activate_from_uploaded_files([uploaded_pdf])
    results = manager.get_retriever().retrieve("ocrtokenviet99 la gi?", top_k=3)

    assert indexed > 0
    assert manager.get_active_source() == "uploaded"
    assert results
    assert any(item.metadata.get("block_type") == "ocr_text" for item in results)
    assert any("ocrtokenviet99" in item.content for item in results)
    activation_stats = manager.get_last_activation_stats()
    assert activation_stats["ocr_chunks"] >= 1


def test_runtime_uploaded_retrieval_filters_to_active_document_ids(tmp_path: Path) -> None:
    provider = _CountingEmbeddingProvider(dimension=8)
    manager = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
        embedding_provider=provider,
    )

    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("alpha-only-token-71 appears in this uploaded file.", encoding="utf-8")
    file_b.write_text("beta-only-token-92 appears in this uploaded file.", encoding="utf-8")
    doc_a = build_doc_id(file_a)
    doc_b = build_doc_id(file_b)

    manager.activate_from_uploaded_files([file_a, file_b], active_document_ids={doc_a})
    results = manager.get_retriever().retrieve("beta-only-token-92", top_k=5)

    assert manager.get_active_source() == "uploaded"
    assert manager.get_active_uploaded_document_ids() == {doc_a}
    assert all(item.doc_id == doc_a for item in results)
    assert all(item.doc_id != doc_b for item in results)


def test_clear_uploaded_indexes_preserves_seeded_runtime_state(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "seed.txt").write_text("seed-token-clear-check-55", encoding="utf-8")

    provider = _CountingEmbeddingProvider(dimension=8)
    manager = RuntimeIndexManager(
        corpus_dir=corpus_dir,
        index_dir=tmp_path / "indexes",
        embedding_provider=provider,
    )

    manager.activate_from_seeded_corpus()
    seeded_results_before = manager.get_retriever().retrieve("seed-token-clear-check-55", top_k=3)
    assert seeded_results_before
    assert manager.get_active_source() == "seeded"

    deleted_files = manager.clear_uploaded_indexes()
    seeded_results_after = manager.get_retriever().retrieve("seed-token-clear-check-55", top_k=3)

    assert deleted_files >= 0
    assert manager.get_active_source() == "seeded"
    assert seeded_results_after


def test_runtime_chunk_strategy_update_changes_chunk_count(tmp_path: Path) -> None:
    provider = _CountingEmbeddingProvider(dimension=8)
    manager = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
        embedding_provider=provider,
        chunk_size=120,
        chunk_overlap=20,
    )

    uploaded = tmp_path / "long.txt"
    uploaded.write_text(
        ("Self-RAG grounding check token. " * 120).strip(),
        encoding="utf-8",
    )

    small_chunk_count = manager.activate_from_uploaded_files([uploaded])
    assert small_chunk_count > 1

    manager.set_chunking(chunk_size=800, chunk_overlap=80)
    large_chunk_count = manager.activate_from_uploaded_files([uploaded])

    assert manager.chunk_size == 800
    assert manager.chunk_overlap == 80
    assert large_chunk_count > 0
    assert large_chunk_count < small_chunk_count


def test_uploaded_manifest_allows_loading_persisted_uploaded_indexes_without_rebuild(tmp_path: Path) -> None:
    provider_first = _CountingEmbeddingProvider(dimension=8)
    manager_first = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
        embedding_provider=provider_first,
    )

    uploaded = tmp_path / "manifest-doc.txt"
    uploaded.write_text("manifest-match-token-6066 appears in uploaded document.", encoding="utf-8")
    doc_id = build_doc_id(uploaded)

    first_count = manager_first.activate_from_uploaded_files([uploaded], active_document_ids={doc_id})
    assert first_count > 0
    manifest_path = tmp_path / "indexes" / "uploaded_index_manifest.json"
    assert manifest_path.exists()

    provider_second = _NoRebuildEmbeddingProvider(dimension=8)
    manager_second = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
        embedding_provider=provider_second,
    )
    loaded_count = manager_second.activate_from_uploaded_files([uploaded], active_document_ids={doc_id})

    assert loaded_count == first_count
    assert manager_second.get_active_source() == "uploaded"
    assert manager_second.get_active_uploaded_document_ids() == {doc_id}
    results = manager_second.get_retriever().retrieve("manifest-match-token-6066", top_k=3)
    assert results
    assert all(item.doc_id == doc_id for item in results)


def test_uploaded_manifest_mismatch_triggers_rebuild_from_ready_documents(tmp_path: Path) -> None:
    provider_first = _CountingEmbeddingProvider(dimension=8)
    manager_first = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
        embedding_provider=provider_first,
    )

    uploaded = tmp_path / "manifest-mismatch.txt"
    uploaded.write_text("manifest-mismatch-token-8282", encoding="utf-8")
    doc_id = build_doc_id(uploaded)
    assert manager_first.activate_from_uploaded_files([uploaded], active_document_ids={doc_id}) > 0

    manifest_path = tmp_path / "indexes" / "uploaded_index_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fingerprint"] = "tampered-fingerprint"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    provider_second = _CountingEmbeddingProvider(dimension=8)
    manager_second = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
        embedding_provider=provider_second,
    )
    rebuilt_count = manager_second.activate_from_uploaded_files([uploaded], active_document_ids={doc_id})

    assert rebuilt_count > 0
    assert provider_second.document_inputs
    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_after["fingerprint"] != "tampered-fingerprint"

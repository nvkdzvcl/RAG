"""API tests for document upload and processing status routes."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


@pytest.fixture
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    data_dir = tmp_path / "data"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "seed.txt").write_text(
        "This seeded corpus text should be replaced by uploaded indexes when available.",
        encoding="utf-8",
    )

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("INDEX_DIR", str(data_dir / "indexes"))
    monkeypatch.setenv("CORPUS_DIR", str(corpus_dir))
    get_settings.cache_clear()

    app = create_app()
    client = TestClient(app)
    try:
        yield client, data_dir
    finally:
        client.close()
        get_settings.cache_clear()


@pytest.fixture
def isolated_client_ocr_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Path]:
    data_dir = tmp_path / "data"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "seed.txt").write_text(
        "Seeded corpus for OCR-enabled upload fixture.",
        encoding="utf-8",
    )

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

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("INDEX_DIR", str(data_dir / "indexes"))
    monkeypatch.setenv("CORPUS_DIR", str(corpus_dir))
    monkeypatch.setenv("OCR_ENABLED", "true")
    monkeypatch.setenv("OCR_LANGUAGE", "vie+eng")
    monkeypatch.setenv("OCR_MIN_TEXT_CHARS", "100")
    monkeypatch.setenv("OCR_RENDER_DPI", "216")
    monkeypatch.setenv("OCR_CONFIDENCE_THRESHOLD", "40")
    monkeypatch.setenv("TESSERACT_CMD", "")
    monkeypatch.setattr(
        "app.ingestion.parsers.pdf_parser.pdfplumber",
        SimpleNamespace(open=lambda _: FakePDF()),
    )
    monkeypatch.setattr("app.ingestion.parsers.pdf_parser.is_tesseract_available", lambda: True)
    monkeypatch.setattr(
        "app.ingestion.parsers.pdf_parser.ocr_pdf_page_with_pymupdf",
        lambda *args, **kwargs: "Nội dung OCR tiếng Việt: tokendebugocr-77",
    )
    get_settings.cache_clear()

    app = create_app()
    client = TestClient(app)
    try:
        yield client, data_dir
    finally:
        client.close()
        get_settings.cache_clear()


def test_upload_endpoint_returns_document_payload(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, data_dir = isolated_client

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("uploaded.txt", b"Upload flow should build hybrid indexes from this chunk.", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()

    assert body["document_id"]
    assert body["id"] == body["document_id"]
    assert body["filename"] == "uploaded.txt"
    assert body["status"] == "ready"
    assert body["chunk_count"] is not None
    assert body["chunk_count"] > 0

    raw_files = sorted((data_dir / "raw").glob("*"))
    assert raw_files
    assert any(path.name.endswith("_uploaded.txt") for path in raw_files)


def test_upload_response_reports_ocr_debug_metadata(
    isolated_client_ocr_enabled: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client_ocr_enabled

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("testocr.pdf", b"%PDF-1.4 fake scan", "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["total_blocks"] >= 1
    assert body["text_blocks"] >= 1
    assert body["ocr_blocks"] >= 1
    assert body["total_chunks"] >= 1


def test_upload_endpoint_accepts_docx(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client

    from docx import Document

    buffer = BytesIO()
    doc = Document()
    doc.add_heading("Báo cáo", level=1)
    doc.add_paragraph("Tài liệu DOCX có bảng và tiếng Việt.")
    doc.save(buffer)
    buffer.seek(0)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "report.docx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "report.docx"
    assert body["status"] == "ready"


def test_document_status_endpoint_returns_saved_state(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client

    upload_response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "guide.md",
                b"# Guide\n\nRuntime indexing should move this document to ready state.",
                "text/markdown",
            )
        },
    )
    assert upload_response.status_code == 201
    document_id = upload_response.json()["document_id"]

    response = client.get(f"/api/v1/documents/{document_id}/status")
    assert response.status_code == 200

    body = response.json()
    assert body["document_id"] == document_id
    assert body["filename"] == "guide.md"
    assert body["status"] == "ready"
    assert body["chunk_count"] is not None

    listed = client.get("/api/v1/documents")
    assert listed.status_code == 200
    listed_body = listed.json()
    assert "documents" in listed_body
    assert any(item["document_id"] == document_id for item in listed_body["documents"])


def test_query_uses_uploaded_document_chunks_when_available(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client

    upload_response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "knowledge.txt",
                b"Blue-ocean sentinel token: nimbus-42. This phrase is unique to uploaded docs.",
                "text/plain",
            )
        },
    )
    assert upload_response.status_code == 201
    uploaded_document_id = upload_response.json()["document_id"]

    query_response = client.post(
        "/api/v1/query",
        json={
            "query": "What does nimbus-42 refer to?",
            "mode": "standard",
            "chat_history": [],
        },
    )
    assert query_response.status_code == 200

    body = query_response.json()
    citations = body["citations"]
    assert citations
    assert any(citation["doc_id"] == uploaded_document_id for citation in citations)
    assert body["trace"][0]["index_source"] == "uploaded"


def test_query_uses_seeded_corpus_when_no_uploaded_documents_ready(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client

    query_response = client.post(
        "/api/v1/query",
        json={
            "query": "What is in the seeded corpus?",
            "mode": "standard",
            "chat_history": [],
        },
    )
    assert query_response.status_code == 200

    body = query_response.json()
    assert body["trace"]
    assert body["trace"][0]["step"] == "retrieve"
    assert body["trace"][0]["index_source"] == "seeded"


def test_standard_mode_retrieves_vietnamese_uploaded_chunks(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client

    upload_response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "vi-knowledge.txt",
                "Hệ thống truy hồi thông tin dùng cụm từ khóa độc nhất: baobietviet42.".encode("utf-8"),
                "text/plain",
            )
        },
    )
    assert upload_response.status_code == 201
    uploaded_document_id = upload_response.json()["document_id"]

    query_response = client.post(
        "/api/v1/query",
        json={
            "query": "baobietviet42 có ý nghĩa gì trong tài liệu?",
            "mode": "standard",
            "chat_history": [],
        },
    )
    assert query_response.status_code == 200

    body = query_response.json()
    citations = body["citations"]
    assert citations
    assert any(citation["doc_id"] == uploaded_document_id for citation in citations)


def test_delete_all_documents_clears_registry_raw_files_and_runtime_indexes(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, data_dir = isolated_client

    first_upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc-a.txt", b"alpha token xoa-all-001", "text/plain")},
    )
    second_upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("doc-b.txt", b"beta token xoa-all-002", "text/plain")},
    )
    assert first_upload.status_code == 201
    assert second_upload.status_code == 201

    listed_before = client.get("/api/v1/documents")
    assert listed_before.status_code == 200
    assert len(listed_before.json()["documents"]) == 2

    delete_response = client.delete("/api/v1/documents")
    assert delete_response.status_code == 200
    payload = delete_response.json()
    assert payload["status"] == "deleted"
    assert payload["deleted_documents"] == 2
    assert payload["deleted_files"] >= 2

    listed_after = client.get("/api/v1/documents")
    assert listed_after.status_code == 200
    assert listed_after.json()["documents"] == []

    raw_files = [path for path in (data_dir / "raw").rglob("*") if path.is_file()]
    assert raw_files == []


def test_delete_all_documents_when_empty_is_idempotent(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client

    response = client.delete("/api/v1/documents")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "deleted"
    assert payload["deleted_documents"] == 0
    assert payload["deleted_files"] == 0


def test_delete_single_document_rebuilds_runtime_from_remaining_uploaded_docs(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, data_dir = isolated_client

    first_upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("first.txt", b"first doc token remove-this-doc", "text/plain")},
    )
    second_upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("second.txt", b"second doc token keep-this-doc-999", "text/plain")},
    )
    assert first_upload.status_code == 201
    assert second_upload.status_code == 201
    first_id = first_upload.json()["document_id"]
    second_id = second_upload.json()["document_id"]

    delete_response = client.delete(f"/api/v1/documents/{first_id}")
    assert delete_response.status_code == 200
    payload = delete_response.json()
    assert payload["status"] == "deleted"
    assert payload["document_id"] == first_id
    assert payload["remaining_documents"] == 1
    assert payload["deleted_files"] >= 1

    listed = client.get("/api/v1/documents")
    assert listed.status_code == 200
    listed_ids = [item["document_id"] for item in listed.json()["documents"]]
    assert first_id not in listed_ids
    assert second_id in listed_ids

    raw_files = [path.name for path in (data_dir / "raw").rglob("*") if path.is_file()]
    assert len(raw_files) == 1

    query_response = client.post(
        "/api/v1/query",
        json={
            "query": "keep-this-doc-999 có trong tài liệu nào?",
            "mode": "standard",
            "chat_history": [],
        },
    )
    assert query_response.status_code == 200
    body = query_response.json()
    assert body["trace"][0]["index_source"] == "uploaded"
    assert any(citation["doc_id"] == second_id for citation in body["citations"])


def test_delete_missing_document_returns_404(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client

    response = client.delete("/api/v1/documents/missing-document-id")
    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]


def test_query_falls_back_to_seeded_after_delete_all_uploaded_documents(
    isolated_client: tuple[TestClient, Path],
) -> None:
    client, _ = isolated_client

    upload_response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "to-delete.txt",
                b"uploaded-doc-token that will be removed before query",
                "text/plain",
            )
        },
    )
    assert upload_response.status_code == 201

    deleted = client.delete("/api/v1/documents")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_documents"] == 1

    query_response = client.post(
        "/api/v1/query",
        json={
            "query": "What is in the seeded corpus?",
            "mode": "standard",
            "chat_history": [],
        },
    )
    assert query_response.status_code == 200
    body = query_response.json()
    assert body["trace"][0]["index_source"] == "seeded"

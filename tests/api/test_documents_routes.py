"""API tests for document upload and processing status routes."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

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

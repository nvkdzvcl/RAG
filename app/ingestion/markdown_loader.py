"""Loader for markdown documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.base_loader import BaseLoader, build_doc_id
from app.schemas.ingestion import LoadedDocument


class MarkdownLoader(BaseLoader):
    """Load .md/.markdown files and split by heading sections."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def load(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LoadedDocument]:
        if not self.supports(path):
            raise ValueError(f"Unsupported markdown file: {path}")

        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()

        file_doc_id = doc_id or build_doc_id(path)
        sections: list[tuple[str | None, str]] = []
        current_section: str | None = None
        buffer: list[str] = []
        extracted_title = title

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                if buffer and "".join(buffer).strip():
                    sections.append((current_section, "\n".join(buffer).strip()))
                heading = stripped.lstrip("#").strip() or None
                if extracted_title is None and stripped.startswith("# ") and heading:
                    extracted_title = heading
                current_section = heading
                buffer = []
            else:
                buffer.append(line)

        if buffer and "".join(buffer).strip():
            sections.append((current_section, "\n".join(buffer).strip()))

        if not sections:
            sections = [(None, raw.strip())]

        base_metadata = dict(metadata or {})
        final_title = extracted_title or path.stem

        documents: list[LoadedDocument] = []
        for section, content in sections:
            if not content:
                continue
            documents.append(
                LoadedDocument(
                    doc_id=file_doc_id,
                    source=str(path),
                    title=final_title,
                    section=section,
                    page=None,
                    content=content,
                    metadata=base_metadata,
                )
            )
        return documents

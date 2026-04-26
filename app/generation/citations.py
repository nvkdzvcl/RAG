"""Citation construction and formatting utilities."""

from __future__ import annotations

from app.schemas.common import Citation
from app.schemas.retrieval import RetrievalResult


class CitationBuilder:
    """Build citation objects from selected retrieval context."""

    def build(self, docs: list[RetrievalResult], max_citations: int = 5) -> list[Citation]:
        if max_citations <= 0:
            return []

        citations: list[Citation] = []
        seen_chunk_ids: set[str] = set()

        for doc in docs:
            if doc.chunk_id in seen_chunk_ids:
                continue
            citations.append(
                Citation(
                    chunk_id=doc.chunk_id,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    section=doc.section,
                    page=doc.page,
                    file_name=str(doc.metadata.get("file_name") or doc.metadata.get("filename"))
                    if (doc.metadata.get("file_name") or doc.metadata.get("filename"))
                    else None,
                    file_type=str(doc.metadata.get("file_type")) if doc.metadata.get("file_type") else None,
                    uploaded_at=str(doc.metadata.get("uploaded_at")) if doc.metadata.get("uploaded_at") else None,
                    created_at=str(doc.metadata.get("created_at")) if doc.metadata.get("created_at") else None,
                    block_type=str(doc.metadata.get("block_type")) if doc.metadata.get("block_type") else None,
                    ocr=bool(doc.metadata.get("ocr")) if "ocr" in doc.metadata else None,
                )
            )
            seen_chunk_ids.add(doc.chunk_id)
            if len(citations) >= max_citations:
                break

        return citations

    def format_citations(self, citations: list[Citation]) -> list[str]:
        """Format citations for display/debug output."""
        lines: list[str] = []
        for idx, citation in enumerate(citations, start=1):
            title = citation.title or "Untitled"
            section = f" | section={citation.section}" if citation.section else ""
            page = f" | page={citation.page}" if citation.page is not None else ""
            lines.append(
                f"[{idx}] {title} | source={citation.source}"
                f" | doc_id={citation.doc_id} | chunk_id={citation.chunk_id}{section}{page}"
            )
        return lines

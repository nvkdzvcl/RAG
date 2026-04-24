"""Chunking logic for ingestion documents."""

from __future__ import annotations

from hashlib import sha1

from app.schemas.ingestion import DocumentChunk, LoadedDocument


class Chunker:
    """Split cleaned documents into overlapping text chunks."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def generate_chunk_id(doc_id: str, chunk_index: int, content: str) -> str:
        """Generate deterministic chunk id from doc and content."""
        digest = sha1(content.encode("utf-8")).hexdigest()[:10]
        return f"{doc_id}_chunk_{chunk_index:04d}_{digest}"

    def chunk_document(self, doc: LoadedDocument) -> list[DocumentChunk]:
        text = doc.content.strip()
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunk_id = self.generate_chunk_id(doc.doc_id, chunk_index, piece)
                metadata = dict(doc.metadata)
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                    }
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        source=doc.source,
                        title=doc.title,
                        section=doc.section,
                        page=doc.page,
                        content=piece,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break
            start = end - self.chunk_overlap

        return chunks

    def chunk_documents(self, docs: list[LoadedDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for doc in docs:
            chunks.extend(self.chunk_document(doc))
        return chunks

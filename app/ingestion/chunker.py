"""Structure-aware chunking logic for ingestion documents."""

from __future__ import annotations

import re
from hashlib import sha1

from app.ingestion.parsers.utils import split_paragraphs
from app.schemas.ingestion import DocumentChunk, LoadedDocument


class Chunker:
    """Split structured documents into token-aware chunks."""

    _token_pattern = re.compile(r"\S+")

    def __init__(self, chunk_size: int = 320, chunk_overlap: int = 40) -> None:
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

    def _token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in self._token_pattern.finditer(text)]

    def _slice_by_token_window(
        self,
        text: str,
        token_spans: list[tuple[int, int]],
        start_token: int,
        end_token: int,
    ) -> str:
        start_char = token_spans[start_token][0]
        end_char = token_spans[end_token - 1][1]
        return text[start_char:end_char].strip()

    def _make_chunk(
        self,
        doc: LoadedDocument,
        *,
        chunk_index: int,
        content: str,
        token_start: int | None,
        token_end: int | None,
    ) -> DocumentChunk:
        chunk_id = self.generate_chunk_id(doc.doc_id, chunk_index, content)
        metadata = dict(doc.metadata)
        metadata.update(
            {
                "chunk_index": chunk_index,
                "token_start": token_start,
                "token_end": token_end,
                "block_type": doc.block_type,
                "language": doc.language,
            }
        )
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            section=doc.section,
            page=doc.page,
            content=content,
            block_type=doc.block_type,
            language=doc.language,
            metadata=metadata,
        )

    def _chunk_text_content(self, doc: LoadedDocument, chunk_index_start: int) -> list[DocumentChunk]:
        paragraphs = split_paragraphs(doc.content)
        if not paragraphs:
            paragraphs = [doc.content]

        chunks: list[DocumentChunk] = []
        chunk_index = chunk_index_start

        for paragraph in paragraphs:
            paragraph_text = paragraph.strip()
            if not paragraph_text:
                continue

            token_spans = self._token_spans(paragraph_text)
            if not token_spans:
                continue

            if len(token_spans) <= self.chunk_size:
                chunks.append(
                    self._make_chunk(
                        doc,
                        chunk_index=chunk_index,
                        content=paragraph_text,
                        token_start=0,
                        token_end=len(token_spans),
                    )
                )
                chunk_index += 1
                continue

            start_token = 0
            while start_token < len(token_spans):
                end_token = min(start_token + self.chunk_size, len(token_spans))
                piece = self._slice_by_token_window(
                    paragraph_text,
                    token_spans,
                    start_token,
                    end_token,
                )
                if piece:
                    chunks.append(
                        self._make_chunk(
                            doc,
                            chunk_index=chunk_index,
                            content=piece,
                            token_start=start_token,
                            token_end=end_token,
                        )
                    )
                    chunk_index += 1

                if end_token >= len(token_spans):
                    break
                start_token = max(end_token - self.chunk_overlap, start_token + 1)

        return chunks

    def chunk_document(self, doc: LoadedDocument) -> list[DocumentChunk]:
        content = doc.content.strip()
        if not content:
            return []

        if doc.block_type in {"table", "image"}:
            token_count = len(self._token_spans(content))
            return [
                self._make_chunk(
                    doc,
                    chunk_index=0,
                    content=content,
                    token_start=0,
                    token_end=token_count if token_count else None,
                )
            ]

        return self._chunk_text_content(doc, 0)

    def chunk_documents(self, docs: list[LoadedDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for doc in docs:
            chunks.extend(self.chunk_document(doc))
        return chunks

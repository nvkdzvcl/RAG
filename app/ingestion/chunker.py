"""Structure-aware chunking logic for ingestion documents."""

from __future__ import annotations

import re
from hashlib import sha1
from pathlib import Path

from app.ingestion.parsers.utils import split_paragraphs
from app.schemas.ingestion import DocumentChunk, LoadedDocument


class Chunker:
    """Split structured documents into token-aware chunks."""

    _token_pattern = re.compile(r"\S+")

    def __init__(
        self,
        chunk_size: int = 320,
        chunk_overlap: int = 40,
        include_heading_context: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_heading_context = include_heading_context

    @staticmethod
    def generate_chunk_id(doc_id: str, chunk_index: int, content: str) -> str:
        """Generate deterministic chunk id from doc and content."""
        digest = sha1(content.encode("utf-8")).hexdigest()[:10]
        return f"{doc_id}_chunk_{chunk_index:04d}_{digest}"

    def _token_spans(self, text: str) -> list[tuple[int, int]]:
        return [
            (match.start(), match.end()) for match in self._token_pattern.finditer(text)
        ]

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

    def _build_heading_prefix(self, doc: LoadedDocument) -> str:
        if not self.include_heading_context:
            return ""

        title = (doc.title or "").strip()
        section = (doc.section or "").strip()

        if len(title) > 100:
            title = title[:97] + "..."
        if len(section) > 100:
            section = section[:97] + "..."

        parts = []
        if title:
            parts.append(f"Title: {title}")
        if section and section != title:
            parts.append(f"Section: {section}")

        if parts:
            return f"[{' | '.join(parts)}]\n"
        return ""

    def _make_chunk(
        self,
        doc: LoadedDocument,
        *,
        chunk_index: int,
        content: str,
        token_start: int | None,
        token_end: int | None,
        heading_prefix: str = "",
    ) -> DocumentChunk:
        final_content = content
        injected = False
        if heading_prefix and not content.startswith(heading_prefix):
            final_content = f"{heading_prefix}\n{content}".strip()
            injected = True

        chunk_id = self.generate_chunk_id(doc.doc_id, chunk_index, final_content)
        metadata = dict(doc.metadata)
        fallback_file_name = Path(doc.source).name
        file_extension = str(
            metadata.get("file_extension") or Path(doc.source).suffix.lower()
        )
        file_type = str(metadata.get("file_type") or file_extension.lstrip("."))
        source_block_type = str(metadata.get("block_type") or doc.block_type)
        metadata.update(
            {
                "chunk_index": chunk_index,
                "token_start": token_start,
                "token_end": token_end,
                "doc_id": doc.doc_id,
                "file_name": str(
                    metadata.get("file_name")
                    or metadata.get("filename")
                    or fallback_file_name
                ),
                "filename": str(
                    metadata.get("filename")
                    or metadata.get("file_name")
                    or fallback_file_name
                ),
                "file_extension": file_extension,
                "file_type": file_type,
                "page": doc.page,
                "block_type": source_block_type,
                "ocr": bool(metadata.get("ocr", False)),
                "language": doc.language,
                "heading_context_injected": injected,
            }
        )
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            section=doc.section,
            page=doc.page,
            content=final_content,
            block_type=doc.block_type,
            language=doc.language,
            metadata=metadata,
        )

    def _chunk_text_content(
        self, doc: LoadedDocument, chunk_index_start: int
    ) -> list[DocumentChunk]:
        heading_prefix = self._build_heading_prefix(doc)
        prefix_tokens = len(self._token_spans(heading_prefix)) if heading_prefix else 0
        effective_chunk_size = max(10, self.chunk_size - prefix_tokens)

        paragraphs = split_paragraphs(doc.content)
        if not paragraphs:
            paragraphs = [doc.content]

        merged_paragraphs: list[str] = []
        current_merge: list[str] = []
        current_tokens = 0

        for p in paragraphs:
            p_text = p.strip()
            if not p_text:
                continue

            spans = self._token_spans(p_text)
            if not spans:
                continue

            p_len = len(spans)

            if current_tokens > 0 and current_tokens + p_len > effective_chunk_size:
                merged_paragraphs.append("\n\n".join(current_merge))
                current_merge = []
                current_tokens = 0

            current_merge.append(p_text)
            current_tokens += p_len

        if current_merge:
            merged_paragraphs.append("\n\n".join(current_merge))

        chunks: list[DocumentChunk] = []
        chunk_index = chunk_index_start

        for paragraph_text in merged_paragraphs:
            token_spans = self._token_spans(paragraph_text)
            if not token_spans:
                continue

            if len(token_spans) <= effective_chunk_size:
                chunks.append(
                    self._make_chunk(
                        doc,
                        chunk_index=chunk_index,
                        content=paragraph_text,
                        token_start=0,
                        token_end=len(token_spans),
                        heading_prefix=heading_prefix,
                    )
                )
                chunk_index += 1
                continue

            start_token = 0
            while start_token < len(token_spans):
                end_token = min(start_token + effective_chunk_size, len(token_spans))
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
                            heading_prefix=heading_prefix,
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
            heading_prefix = self._build_heading_prefix(doc)
            token_count = len(self._token_spans(content))
            return [
                self._make_chunk(
                    doc,
                    chunk_index=0,
                    content=content,
                    token_start=0,
                    token_end=token_count if token_count else None,
                    heading_prefix=heading_prefix,
                )
            ]

        return self._chunk_text_content(doc, 0)

    def _group_text_documents(self, docs: list[LoadedDocument]) -> list[LoadedDocument]:
        if not docs:
            return []

        grouped: list[LoadedDocument] = []
        current_group: list[LoadedDocument] = []

        def flush_group():
            if not current_group:
                return
            if len(current_group) == 1:
                grouped.append(current_group[0])
            else:
                base_doc = current_group[0]
                merged_content = "\n\n".join(
                    d.content.strip() for d in current_group if d.content.strip()
                )
                new_doc = LoadedDocument(
                    doc_id=base_doc.doc_id,
                    source=base_doc.source,
                    title=base_doc.title,
                    section=base_doc.section,
                    page=base_doc.page,
                    content=merged_content,
                    block_type="text",
                    language=base_doc.language,
                    metadata=dict(base_doc.metadata),
                )
                grouped.append(new_doc)
            current_group.clear()

        for doc in docs:
            if doc.block_type != "text":
                flush_group()
                grouped.append(doc)
                continue

            if current_group:
                prev = current_group[-1]
                if doc.doc_id != prev.doc_id or doc.section != prev.section:
                    flush_group()

            current_group.append(doc)

        flush_group()
        return grouped

    def chunk_documents(self, docs: list[LoadedDocument]) -> list[DocumentChunk]:
        grouped_docs = self._group_text_documents(docs)
        chunks: list[DocumentChunk] = []
        # Maintain a chunk_index across the entire processing output to prevent minor hash collisions if chunking very repetitive text
        for doc in grouped_docs:
            chunks.extend(self.chunk_document(doc))
        return chunks

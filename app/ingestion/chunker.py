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
    _sentence_split_pattern = re.compile(r"(?<=[.!?])\s+")
    _prefix_value_max_chars = 72

    def __init__(
        self,
        chunk_size: int = 320,
        chunk_overlap: int = 40,
        include_heading_context: bool = True,
        max_grouped_chars: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if max_grouped_chars is not None and max_grouped_chars <= 0:
            raise ValueError("max_grouped_chars must be positive when provided")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_heading_context = include_heading_context
        # Safety cap to avoid over-merging too many text blocks into one mega document.
        self.max_grouped_chars = max_grouped_chars or max(1200, self.chunk_size * 12)

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

        if len(title) > self._prefix_value_max_chars:
            title = title[: self._prefix_value_max_chars - 3].rstrip() + "..."
        if len(section) > self._prefix_value_max_chars:
            section = section[: self._prefix_value_max_chars - 3].rstrip() + "..."

        parts = []
        if title:
            parts.append(f"Title: {title}.")
        if section and section != title:
            parts.append(f"Section: {section}.")

        if parts:
            return " ".join(parts) + "\n\n"
        return ""

    def _enforce_chunk_budget(
        self, *, content: str, heading_prefix: str
    ) -> tuple[str, bool, int]:
        """Enforce chunk_size after heading prefix injection where possible."""
        if not heading_prefix:
            return content, False, 0

        prefix_tokens = len(self._token_spans(heading_prefix))
        if prefix_tokens >= self.chunk_size:
            # Avoid empty raw-content chunks when prefix consumes the whole budget.
            return content, False, 0

        raw_tokens = self._token_spans(content)
        available_tokens = self.chunk_size - prefix_tokens
        if len(raw_tokens) > available_tokens:
            content = self._slice_by_token_window(
                content,
                raw_tokens,
                0,
                available_tokens,
            )

        return f"{heading_prefix}{content}".strip(), True, len(heading_prefix)

    def _looks_like_heading_paragraph(
        self, paragraph: str, *, heading_hint: bool
    ) -> bool:
        token_count = len(self._token_spans(paragraph))
        if token_count == 0:
            return False
        stripped = paragraph.strip()
        if stripped.startswith("#"):
            return True
        if heading_hint and token_count <= 24:
            return True
        if stripped.endswith(":") and token_count <= 18:
            return True
        if "\n" not in stripped and token_count <= 10 and stripped[:1].isupper():
            return True
        return False

    def _merge_short_paragraphs(
        self,
        paragraphs: list[str],
        effective_chunk_size: int,
        *,
        heading_hint: bool,
    ) -> list[str]:
        """Merge sequential short paragraphs to avoid tiny isolated chunks."""
        normalized = [
            paragraph.strip() for paragraph in paragraphs if paragraph.strip()
        ]
        if not normalized:
            return []

        paragraph_units: list[tuple[str, int]] = []
        for paragraph in normalized:
            token_count = len(self._token_spans(paragraph))
            if token_count > 0:
                paragraph_units.append((paragraph, token_count))
        if not paragraph_units:
            return []

        merged_units: list[tuple[str, int]] = []
        current_parts: list[str] = []
        current_tokens = 0

        for index, (paragraph, token_count) in enumerate(paragraph_units):
            if not current_parts:
                current_parts = [paragraph]
                current_tokens = token_count
                continue

            projected_tokens = current_tokens + token_count
            if projected_tokens <= effective_chunk_size:
                current_parts.append(paragraph)
                current_tokens += token_count
                continue

            # Keep short heading text attached to the following paragraph block.
            if len(current_parts) == 1 and self._looks_like_heading_paragraph(
                current_parts[0],
                heading_hint=heading_hint and index == 1,
            ):
                current_parts.append(paragraph)
                current_tokens = projected_tokens
                continue

            merged_units.append(("\n\n".join(current_parts), current_tokens))
            current_parts = [paragraph]
            current_tokens = token_count

        if current_parts:
            merged_units.append(("\n\n".join(current_parts), current_tokens))

        min_tokens = max(1, min(16, effective_chunk_size // 4))
        stabilized: list[tuple[str, int]] = []
        for text, token_count in merged_units:
            if (
                stabilized
                and token_count < min_tokens
                and stabilized[-1][1] + token_count <= effective_chunk_size
            ):
                prev_text, prev_tokens = stabilized[-1]
                stabilized[-1] = (f"{prev_text}\n\n{text}", prev_tokens + token_count)
                continue
            stabilized.append((text, token_count))

        return [text for text, _ in stabilized]

    def _split_sentence_units(self, text: str) -> list[str]:
        parts = [
            sentence.strip()
            for sentence in self._sentence_split_pattern.split(text.replace("\n", " "))
            if sentence and sentence.strip()
        ]
        return parts or [text.strip()]

    def _apply_sentence_boundary_chunking(
        self, text: str, max_tokens: int
    ) -> list[str] | None:
        """Prefer sentence boundary splits for long paragraphs when possible."""
        sentences = self._split_sentence_units(text)
        if len(sentences) <= 1:
            return None

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self._token_spans(sentence))
            if sentence_tokens == 0:
                continue
            if sentence_tokens > max_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                return None

            projected_tokens = current_tokens + sentence_tokens
            if current_sentences and projected_tokens > max_tokens:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens = projected_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks if len(chunks) > 1 else None

    def _effective_overlap_tokens(self, window_tokens: int) -> int:
        if window_tokens <= 1:
            return 0
        min_stride = max(1, min(8, window_tokens // 4))
        max_overlap = max(0, window_tokens - min_stride)
        return min(self.chunk_overlap, max_overlap)

    def _split_with_overlap_windows(
        self, text: str, window_tokens: int
    ) -> list[tuple[str, int, int]]:
        token_spans = self._token_spans(text)
        if not token_spans:
            return []
        if len(token_spans) <= window_tokens:
            return [(text.strip(), 0, len(token_spans))]

        overlap_tokens = self._effective_overlap_tokens(window_tokens)
        stride = max(1, window_tokens - overlap_tokens)

        chunks: list[tuple[str, int, int]] = []
        seen_ranges: set[tuple[int, int]] = set()
        previous_piece = ""
        start_token = 0
        total_tokens = len(token_spans)

        while start_token < total_tokens:
            end_token = min(start_token + window_tokens, total_tokens)
            if (start_token, end_token) in seen_ranges:
                break
            seen_ranges.add((start_token, end_token))

            piece = self._slice_by_token_window(
                text,
                token_spans,
                start_token,
                end_token,
            )
            if piece and piece != previous_piece:
                chunks.append((piece, start_token, end_token))
                previous_piece = piece

            if end_token >= total_tokens:
                break

            next_start = start_token + stride
            if next_start <= start_token:
                next_start = start_token + 1

            remaining_tokens = total_tokens - next_start
            if remaining_tokens <= 1 and total_tokens > window_tokens:
                break

            start_token = next_start

        return chunks

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
        prefix_length = 0
        if heading_prefix and not content.startswith(heading_prefix):
            final_content, injected, prefix_length = self._enforce_chunk_budget(
                content=content,
                heading_prefix=heading_prefix,
            )

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
                "heading_context_prefix_length": prefix_length,
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
        candidate_prefix = self._build_heading_prefix(doc)
        prefix_tokens = (
            len(self._token_spans(candidate_prefix)) if candidate_prefix else 0
        )
        inject_prefix = bool(candidate_prefix) and prefix_tokens < self.chunk_size
        heading_prefix = candidate_prefix if inject_prefix else ""
        effective_chunk_size = (
            max(1, self.chunk_size - prefix_tokens)
            if inject_prefix
            else self.chunk_size
        )

        paragraphs = split_paragraphs(doc.content)
        if not paragraphs:
            paragraphs = [doc.content]

        merged_paragraphs = self._merge_short_paragraphs(
            paragraphs,
            effective_chunk_size,
            heading_hint=bool(doc.metadata.get("is_heading")),
        )

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

            sentence_split = self._apply_sentence_boundary_chunking(
                paragraph_text, effective_chunk_size
            )
            if sentence_split is not None:
                for sentence_piece in sentence_split:
                    sentence_tokens = self._token_spans(sentence_piece)
                    if not sentence_tokens:
                        continue
                    chunks.append(
                        self._make_chunk(
                            doc,
                            chunk_index=chunk_index,
                            content=sentence_piece,
                            token_start=0,
                            token_end=len(sentence_tokens),
                            heading_prefix=heading_prefix,
                        )
                    )
                    chunk_index += 1
                continue

            windows = self._split_with_overlap_windows(
                paragraph_text, effective_chunk_size
            )
            for piece, start_token, end_token in windows:
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
        current_group_chars = 0

        def flush_group():
            nonlocal current_group_chars
            if not current_group:
                return
            if len(current_group) == 1:
                grouped.append(current_group[0])
            else:
                base_doc = current_group[0]
                merged_content = "\n\n".join(
                    d.content.strip() for d in current_group if d.content.strip()
                )
                merged_metadata = dict(base_doc.metadata)
                for member in current_group[1:]:
                    for key, value in member.metadata.items():
                        if key not in merged_metadata:
                            merged_metadata[key] = value
                if any(
                    bool(member.metadata.get("is_heading")) for member in current_group
                ):
                    merged_metadata["is_heading"] = True
                new_doc = LoadedDocument(
                    doc_id=base_doc.doc_id,
                    source=base_doc.source,
                    title=base_doc.title,
                    section=base_doc.section,
                    page=base_doc.page,
                    content=merged_content,
                    block_type="text",
                    language=base_doc.language,
                    metadata=merged_metadata,
                )
                grouped.append(new_doc)
            current_group.clear()
            current_group_chars = 0

        def _is_heading_doc(doc: LoadedDocument) -> bool:
            if bool(doc.metadata.get("is_heading")):
                return True
            return doc.content.strip().startswith("#")

        def can_group(prev: LoadedDocument, current: LoadedDocument) -> bool:
            if prev.doc_id != current.doc_id:
                return False
            if prev.section == current.section:
                return True
            prev_is_heading = bool(prev.metadata.get("is_heading"))
            prev_section = (prev.section or "").strip()
            current_section = (current.section or "").strip()
            if prev_is_heading and (not prev_section or not current_section):
                return True
            return False

        for doc in docs:
            if doc.block_type != "text":
                flush_group()
                grouped.append(doc)
                continue

            if current_group:
                prev = current_group[-1]
                if not can_group(prev, doc):
                    flush_group()
                else:
                    projected_chars = current_group_chars + len(doc.content.strip())
                    if projected_chars > self.max_grouped_chars:
                        # Keep heading attached to first paragraph when possible.
                        if not (
                            len(current_group) == 1
                            and _is_heading_doc(current_group[0])
                        ):
                            flush_group()

            current_group.append(doc)
            current_group_chars += len(doc.content.strip())

        flush_group()
        return grouped

    def chunk_documents(self, docs: list[LoadedDocument]) -> list[DocumentChunk]:
        grouped_docs = self._group_text_documents(docs)
        chunks: list[DocumentChunk] = []
        # Maintain a chunk_index across the entire processing output to prevent minor hash collisions if chunking very repetitive text
        for doc in grouped_docs:
            chunks.extend(self.chunk_document(doc))
        return chunks

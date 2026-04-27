"""Markdown parser implementation for mixed block extraction."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers.base import BaseDocumentParser
from app.ingestion.parsers.utils import split_paragraphs
from app.schemas.ingestion import DocumentBlock


class MarkdownParser(BaseDocumentParser):
    """Parse markdown into text/table/image blocks with section metadata."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def _flush_text_buffer(
        self, blocks: list[DocumentBlock], buffer: list[str], section: str | None
    ) -> None:
        if not buffer:
            return
        joined = "\n".join(buffer).strip()
        if not joined:
            return
        for paragraph in split_paragraphs(joined):
            blocks.append(
                DocumentBlock(
                    type="text",
                    content=paragraph,
                    metadata={"page": None, "section": section, "bbox": None},
                )
            )
        buffer.clear()

    def _flush_table_buffer(
        self, blocks: list[DocumentBlock], rows: list[str], section: str | None
    ) -> None:
        if not rows:
            return
        table_text = "\n".join(rows).strip()
        if not table_text:
            return
        blocks.append(
            DocumentBlock(
                type="table",
                content=table_text,
                metadata={"page": None, "section": section, "bbox": None},
            )
        )
        rows.clear()

    @staticmethod
    def _is_markdown_table_line(line: str) -> bool:
        stripped = line.strip()
        return "|" in stripped and stripped.count("|") >= 2

    @staticmethod
    def _is_markdown_image_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("![") and "](" in stripped and stripped.endswith(")")

    def parse(self, path: Path) -> list[DocumentBlock]:
        lines = path.read_text(encoding="utf-8").splitlines()
        blocks: list[DocumentBlock] = []
        text_buffer: list[str] = []
        table_buffer: list[str] = []
        current_section: str | None = None

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("#"):
                self._flush_text_buffer(blocks, text_buffer, current_section)
                self._flush_table_buffer(blocks, table_buffer, current_section)
                heading = stripped.lstrip("#").strip()
                current_section = heading or current_section
                if heading:
                    blocks.append(
                        DocumentBlock(
                            type="text",
                            content=heading,
                            metadata={
                                "page": None,
                                "section": current_section,
                                "bbox": None,
                                "is_heading": True,
                            },
                        )
                    )
                continue

            if self._is_markdown_image_line(line):
                self._flush_text_buffer(blocks, text_buffer, current_section)
                self._flush_table_buffer(blocks, table_buffer, current_section)
                blocks.append(
                    DocumentBlock(
                        type="image",
                        content=stripped,
                        metadata={
                            "page": None,
                            "section": current_section,
                            "bbox": None,
                        },
                    )
                )
                continue

            if self._is_markdown_table_line(line):
                self._flush_text_buffer(blocks, text_buffer, current_section)
                table_buffer.append(line.rstrip())
                continue

            if table_buffer and not stripped:
                self._flush_table_buffer(blocks, table_buffer, current_section)
                continue
            if table_buffer and stripped and not self._is_markdown_table_line(line):
                self._flush_table_buffer(blocks, table_buffer, current_section)

            text_buffer.append(line)

        self._flush_text_buffer(blocks, text_buffer, current_section)
        self._flush_table_buffer(blocks, table_buffer, current_section)
        return [block for block in blocks if block.content.strip()]

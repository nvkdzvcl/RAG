"""Text cleaning utilities for ingestion."""

from __future__ import annotations

import re

from app.schemas.ingestion import LoadedDocument


class TextCleaner:
    """Apply light normalization to loaded documents."""

    _multi_blank = re.compile(r"\n{3,}")
    _space_before_newline = re.compile(r"[ \t]+\n")

    def clean_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._space_before_newline.sub("\n", normalized)
        normalized = self._multi_blank.sub("\n\n", normalized)
        return normalized.strip()

    def clean_documents(self, docs: list[LoadedDocument]) -> list[LoadedDocument]:
        cleaned: list[LoadedDocument] = []
        for doc in docs:
            cleaned.append(
                doc.model_copy(update={"content": self.clean_text(doc.content)})
            )
        return cleaned

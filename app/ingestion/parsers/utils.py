"""Shared helpers for parser implementations."""

from __future__ import annotations

import re
from pathlib import Path

TEXT_DECODE_FALLBACK_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "cp1258",
    "latin-1",
)


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraph-like units while preserving UTF-8 content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", normalized)
    return [part.strip() for part in parts if part and part.strip()]


def read_text_with_fallback(path: Path) -> str:
    """Read text using a safe encoding fallback chain."""
    payload = path.read_bytes()
    for encoding in TEXT_DECODE_FALLBACK_ENCODINGS:
        try:
            return payload.decode(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode(encoding="latin-1")


def rows_to_markdown_table(rows: list[list[str | None]]) -> str:
    """Serialize tabular rows into markdown format."""
    if not rows:
        return ""

    normalized_rows: list[list[str]] = []
    max_cols = max(len(row) for row in rows) if rows else 0
    for row in rows:
        normalized_row = [(cell or "").strip() for cell in row]
        if len(normalized_row) < max_cols:
            normalized_row.extend([""] * (max_cols - len(normalized_row)))
        normalized_rows.append(normalized_row)

    if not normalized_rows:
        return ""

    header = normalized_rows[0]
    divider = ["---"] * len(header)
    lines = [
        f"| {' | '.join(header)} |",
        f"| {' | '.join(divider)} |",
    ]
    for row in normalized_rows[1:]:
        lines.append(f"| {' | '.join(row)} |")
    return "\n".join(lines)

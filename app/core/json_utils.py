"""Robust JSON extraction/parsing helpers for LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCED_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", flags=re.IGNORECASE | re.DOTALL
)


def _collect_candidates(raw_text: str) -> list[str]:
    text = raw_text.strip()
    if not text:
        return []

    candidates: list[str] = []

    for block in _FENCED_BLOCK_PATTERN.findall(text):
        cleaned = block.strip()
        if cleaned:
            candidates.append(cleaned)

    candidates.append(text)

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1].strip())

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        candidates.append(text[array_start : array_end + 1].strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def parse_json_payload(raw_text: str) -> Any | None:
    """Parse JSON payload from raw model output, returning None on failure."""
    for candidate in _collect_candidates(raw_text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    """Parse and return a JSON object from model output when available."""
    parsed = parse_json_payload(raw_text)
    if isinstance(parsed, dict):
        return parsed
    return None


def parse_json_list(raw_text: str) -> list[Any] | None:
    """Parse and return a JSON array from model output when available."""
    parsed = parse_json_payload(raw_text)
    if isinstance(parsed, list):
        return parsed
    return None

"""Prompt template loading/rendering utilities."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template


class PromptRepository:
    """Load and render prompt templates from the configured prompt directory."""

    def __init__(self, prompt_dir: str | Path) -> None:
        self.prompt_dir = Path(prompt_dir)
        self._cache: dict[str, str] = {}

    def get(self, file_name: str, *, fallback: str = "") -> str:
        """Return prompt text from file, or fallback when file is missing/empty."""
        cached = self._cache.get(file_name)
        if cached is not None:
            return cached

        path = self.prompt_dir / file_name
        text = ""
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8").strip()

        resolved = text if text else fallback
        self._cache[file_name] = resolved
        return resolved

    def render(self, file_name: str, *, fallback: str = "", **variables: object) -> str:
        """Render prompt template with best-effort variable substitution."""
        template_text = self.get(file_name, fallback=fallback)
        template = Template(template_text)
        normalized = {
            key: self._coerce_value(value) for key, value in variables.items()
        }
        return template.safe_substitute(normalized)

    @staticmethod
    def _coerce_value(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

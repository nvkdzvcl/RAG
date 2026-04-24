"""Query rewrite logic for advanced retrieval retries."""

from __future__ import annotations

from app.schemas.workflow import CritiqueResult


class QueryRewriter:
    """Generate rewritten queries for retry loops."""

    def rewrite(
        self,
        query: str,
        *,
        critique: CritiqueResult | None = None,
        loop_count: int = 0,
    ) -> list[str]:
        base = query.strip()
        candidates: list[str] = []

        if critique and critique.better_queries:
            candidates.extend(critique.better_queries)

        candidates.append(f"{base} with grounded evidence")
        candidates.append(f"{base} with source citations")

        if critique and critique.missing_aspects:
            for aspect in critique.missing_aspects[:2]:
                candidates.append(f"{base} focus on {aspect}")

        if "force retry" in base.lower():
            candidates.append(f"{base} retry iteration {loop_count}")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            cleaned = item.strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            deduped.append(cleaned)
            seen.add(key)
        return deduped

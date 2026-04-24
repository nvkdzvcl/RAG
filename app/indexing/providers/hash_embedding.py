"""Deterministic hash-based embedding provider for local development/tests."""

from __future__ import annotations

import math
import re
from hashlib import sha1


class HashEmbeddingProvider:
    """Simple deterministic embedding provider with no external model dependency."""

    token_pattern = re.compile(r"\w+")

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.name = "hash-embedding"

    def _tokenize(self, text: str) -> list[str]:
        return self.token_pattern.findall(text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            hashed = int(sha1(token.encode("utf-8")).hexdigest(), 16)
            index = hashed % self.dimension
            sign = -1.0 if ((hashed >> 1) & 1) else 1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

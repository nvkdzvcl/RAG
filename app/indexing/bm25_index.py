"""BM25 sparse index implementation."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from app.schemas.ingestion import DocumentChunk

_LEGACY_TOKEN_PATTERN = re.compile(r"\w+")

try:
    from underthesea import word_tokenize as _UNDERTHESEA_WORD_TOKENIZE
except Exception:  # pragma: no cover - optional dependency
    _UNDERTHESEA_WORD_TOKENIZE = None


def _legacy_tokenize(text: str) -> list[str]:
    return _LEGACY_TOKEN_PATTERN.findall(text.lower())


def _tokenize_with_underthesea(text: str) -> list[str]:
    if _UNDERTHESEA_WORD_TOKENIZE is None:
        return []

    try:
        segmented = _UNDERTHESEA_WORD_TOKENIZE(text, format="text")
    except TypeError:
        segmented = _UNDERTHESEA_WORD_TOKENIZE(text)
    except Exception:
        return []

    if isinstance(segmented, str):
        return _LEGACY_TOKEN_PATTERN.findall(segmented.lower())

    tokens: list[str] = []
    for raw in segmented:
        if not isinstance(raw, str):
            continue
        token = raw.strip().lower()
        if not token:
            continue
        tokens.append(token.replace(" ", "_"))
    return tokens


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text for BM25, preferring Vietnamese word segmentation when available."""
    normalized = text.strip().lower()
    if not normalized:
        return []

    segmented = _tokenize_with_underthesea(normalized)
    if segmented:
        return segmented
    return _legacy_tokenize(normalized)


class BM25Index:
    """Pure-Python BM25 index for chunk-level sparse retrieval."""

    token_pattern = _LEGACY_TOKEN_PATTERN

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._chunks: list[DocumentChunk] = []
        self._term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_doc_len: float = 0.0

    @property
    def doc_count(self) -> int:
        return len(self._chunks)

    @property
    def chunk_ids(self) -> list[str]:
        return [chunk.chunk_id for chunk in self._chunks]

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self._chunks

    @property
    def term_freqs(self) -> list[dict[str, int]]:
        return self._term_freqs

    @property
    def doc_lengths(self) -> list[int]:
        return self._doc_lengths

    @property
    def idf(self) -> dict[str, float]:
        return self._idf

    @property
    def avg_doc_len(self) -> float:
        return self._avg_doc_len

    def _tokenize(self, text: str) -> list[str]:
        return tokenize_bm25(text)

    def build(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunks")

        self._chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        self._term_freqs = []
        self._doc_lengths = []

        document_frequency: Counter[str] = Counter()

        for chunk in self._chunks:
            tokens = self._tokenize(chunk.content)
            tf = Counter(tokens)
            self._term_freqs.append(dict(tf))
            doc_len = len(tokens)
            self._doc_lengths.append(doc_len)
            for term in tf:
                document_frequency[term] += 1

        total_length = sum(self._doc_lengths)
        self._avg_doc_len = total_length / len(self._doc_lengths) if self._doc_lengths else 0.0

        doc_count = len(self._chunks)
        self._idf = {}
        for term, freq in document_frequency.items():
            # Positive variant of BM25 IDF.
            self._idf[term] = math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "chunks": [chunk.model_dump() for chunk in self._chunks],
            "term_freqs": self._term_freqs,
            "doc_lengths": self._doc_lengths,
            "idf": self._idf,
            "avg_doc_len": self._avg_doc_len,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BM25Index":
        index = cls(k1=float(payload.get("k1", 1.5)), b=float(payload.get("b", 0.75)))
        index._chunks = [DocumentChunk.model_validate(item) for item in payload.get("chunks", [])]
        index._term_freqs = [dict(freq) for freq in payload.get("term_freqs", [])]
        index._doc_lengths = [int(length) for length in payload.get("doc_lengths", [])]
        index._idf = {str(key): float(value) for key, value in payload.get("idf", {}).items()}
        index._avg_doc_len = float(payload.get("avg_doc_len", 0.0))
        return index

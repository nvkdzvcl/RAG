"""Lightweight thread-safe LRU cache for RAG pipeline components."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CacheStats:
    """Immutable snapshot of cache statistics."""

    hits: int
    misses: int
    size: int
    maxsize: int
    enabled: bool

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0


class QueryCache:
    """Thread-safe LRU cache with an enable/disable switch.

    Parameters
    ----------
    maxsize:
        Maximum number of entries. ``0`` disables the cache regardless of
        ``enabled``.
    enabled:
        Master toggle.  When ``False`` the cache always misses and never
        stores entries.
    """

    def __init__(self, maxsize: int = 128, *, enabled: bool = True) -> None:
        self._maxsize = max(0, maxsize)
        self._enabled = enabled and self._maxsize > 0
        self._lock = threading.Lock()
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get(self, key: str) -> tuple[bool, Any]:
        """Return ``(True, value)`` on hit, ``(False, None)`` on miss."""
        if not self._enabled:
            self._misses += 1
            return False, None

        with self._lock:
            value = self._store.get(key)
            if value is None and key not in self._store:
                self._misses += 1
                return False, None
            self._store.move_to_end(key)
            self._hits += 1
            return True, value

    def put(self, key: str, value: Any) -> None:
        """Insert or update *key*.  Evicts LRU entry when full."""
        if not self._enabled:
            return

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                return
            if len(self._store) >= self._maxsize:
                self._store.popitem(last=False)
            self._store[key] = value

    def invalidate(self) -> None:
        """Drop all cached entries (e.g. after an index rebuild)."""
        with self._lock:
            self._store.clear()

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size=len(self._store),
            maxsize=self._maxsize,
            enabled=self._enabled,
        )

    def __len__(self) -> int:
        return len(self._store)


# ------------------------------------------------------------------
# Key helpers
# ------------------------------------------------------------------


def make_cache_key(*parts: str | int | float | None) -> str:
    """Build a deterministic cache key from arbitrary parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


@dataclass
class CacheGroup:
    """Bundle of caches for pipeline stages."""

    embedding: QueryCache = field(
        default_factory=lambda: QueryCache(maxsize=0, enabled=False)
    )
    retrieval: QueryCache = field(
        default_factory=lambda: QueryCache(maxsize=0, enabled=False)
    )
    llm: QueryCache = field(
        default_factory=lambda: QueryCache(maxsize=0, enabled=False)
    )
    rerank: QueryCache = field(
        default_factory=lambda: QueryCache(maxsize=0, enabled=False)
    )

    def invalidate_all(self) -> None:
        """Clear every cache (e.g. on index rebuild)."""
        self.embedding.invalidate()
        self.retrieval.invalidate()
        self.llm.invalidate()
        self.rerank.invalidate()

    def stats_dict(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding.stats().__dict__,
            "retrieval": self.retrieval.stats().__dict__,
            "llm": self.llm.stats().__dict__,
            "rerank": self.rerank.stats().__dict__,
        }


def create_cache_group_from_settings(settings: Any) -> CacheGroup:
    """Build caches using application settings with safe defaults."""
    cache_enabled = bool(getattr(settings, "cache_enabled", False))

    def _enabled(name: str, default: bool = True) -> bool:
        return cache_enabled and bool(getattr(settings, name, default))

    return CacheGroup(
        embedding=QueryCache(
            maxsize=int(getattr(settings, "cache_embedding_maxsize", 256)),
            enabled=_enabled("embedding_cache_enabled"),
        ),
        retrieval=QueryCache(
            maxsize=int(getattr(settings, "cache_retrieval_maxsize", 128)),
            enabled=_enabled("retrieval_cache_enabled"),
        ),
        llm=QueryCache(
            maxsize=int(getattr(settings, "cache_llm_maxsize", 64)),
            enabled=_enabled("llm_cache_enabled"),
        ),
        rerank=QueryCache(
            maxsize=int(getattr(settings, "cache_rerank_maxsize", 128)),
            enabled=_enabled("rerank_cache_enabled"),
        ),
    )

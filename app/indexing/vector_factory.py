"""Factory helpers for selecting vector index backend implementation."""

from __future__ import annotations

import logging
from typing import Any

from app.indexing.vector_index import InMemoryVectorIndex, VectorIndex

logger = logging.getLogger(__name__)


def create_vector_index(settings: Any) -> VectorIndex:
    """Create vector index instance based on configured backend."""
    backend = str(getattr(settings, "vector_index_backend", "inmemory")).strip().lower()

    if backend == "faiss":
        # Keep faiss dependency isolated until backend selection explicitly requests it.
        from app.indexing.faiss_index import FaissVectorIndex

        return FaissVectorIndex()

    if backend != "inmemory":
        logger.warning(
            "Unsupported vector index backend '%s'. Falling back to inmemory.", backend
        )
    return InMemoryVectorIndex()

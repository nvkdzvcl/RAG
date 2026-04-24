"""Factory helpers for selecting and initializing embedding providers."""

from __future__ import annotations

import logging

from app.indexing.embeddings import BaseEmbeddingProvider
from app.indexing.providers.hash_embedding import HashEmbeddingProvider
from app.indexing.providers.sentence_transformer_embedding import SentenceTransformerEmbeddingProvider

logger = logging.getLogger(__name__)

SENTENCE_TRANSFORMER_PROVIDER_NAMES = {
    "sentence_transformers",
    "sentence-transformers",
    "sentence-transformer",
    "st",
}
HASH_PROVIDER_NAMES = {
    "hash",
    "hash_embedding",
    "hash-embedding",
}


def create_embedding_provider(
    *,
    provider_name: str,
    model: str,
    device: str,
    batch_size: int,
    normalize: bool,
    fallback_hash_dimension: int = 64,
) -> BaseEmbeddingProvider:
    """Create embedding provider with safe fallback to hash embeddings."""
    normalized = provider_name.strip().lower() if provider_name else ""

    if normalized in SENTENCE_TRANSFORMER_PROVIDER_NAMES:
        try:
            return SentenceTransformerEmbeddingProvider(
                model_name=model,
                device=device,
                batch_size=batch_size,
                normalize=normalize,
            )
        except Exception as exc:
            logger.warning(
                (
                    "Failed to initialize sentence-transformers model '%s' on device '%s'. "
                    "Falling back to hash embeddings."
                ),
                model,
                device,
                exc_info=exc,
            )
            return HashEmbeddingProvider(dimension=fallback_hash_dimension)

    if normalized in HASH_PROVIDER_NAMES:
        return HashEmbeddingProvider(dimension=fallback_hash_dimension)

    logger.warning(
        "Unknown embedding provider '%s'. Falling back to hash embeddings.",
        provider_name,
    )
    return HashEmbeddingProvider(dimension=fallback_hash_dimension)

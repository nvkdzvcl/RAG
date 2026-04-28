"""Indexing package with embedding/vector/sparse index components."""

from app.indexing.bm25_index import BM25Index
from app.indexing.builder import BuiltIndexes, IndexBuilder
from app.indexing.embeddings import BaseEmbeddingProvider, EmbeddingProvider
from app.indexing.persistence import LocalIndexStore
from app.indexing.providers.factory import create_embedding_provider
from app.indexing.providers.hash_embedding import HashEmbeddingProvider
from app.indexing.providers.sentence_transformer_embedding import (
    SentenceTransformerEmbeddingProvider,
)
from app.indexing.vector_index import InMemoryVectorIndex, VectorIndex

__all__ = [
    "BaseEmbeddingProvider",
    "BM25Index",
    "BuiltIndexes",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "InMemoryVectorIndex",
    "IndexBuilder",
    "LocalIndexStore",
    "SentenceTransformerEmbeddingProvider",
    "VectorIndex",
    "create_embedding_provider",
]

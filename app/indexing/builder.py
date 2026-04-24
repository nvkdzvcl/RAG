"""Build vector and BM25 indexes from ingestion chunks."""

from __future__ import annotations

from dataclasses import dataclass

from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import EmbeddingProvider
from app.indexing.vector_index import InMemoryVectorIndex, VectorIndex
from app.schemas.ingestion import DocumentChunk


@dataclass
class BuiltIndexes:
    """Container for built indexing artifacts."""

    vector_index: VectorIndex
    bm25_index: BM25Index
    chunk_count: int
    embedding_provider: str


class IndexBuilder:
    """Build both dense and sparse indexes from ingestion output chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index or InMemoryVectorIndex()
        self.bm25_index = bm25_index or BM25Index()

    def build(self, chunks: list[DocumentChunk]) -> BuiltIndexes:
        if not chunks:
            raise ValueError("Cannot build indexes from empty chunks")

        vectors = self.embedding_provider.embed_documents([chunk.content for chunk in chunks])
        self.vector_index.build(chunks, vectors)
        self.bm25_index.build(chunks)

        return BuiltIndexes(
            vector_index=self.vector_index,
            bm25_index=self.bm25_index,
            chunk_count=len(chunks),
            embedding_provider=self.embedding_provider.name,
        )

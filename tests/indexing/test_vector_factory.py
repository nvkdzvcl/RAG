"""Tests for vector index backend factory selection."""

from __future__ import annotations

from types import SimpleNamespace

from app.indexing.faiss_index import FaissVectorIndex
from app.indexing.vector_factory import create_vector_index
from app.indexing.vector_index import InMemoryVectorIndex


def test_create_vector_index_defaults_to_inmemory_when_backend_missing() -> None:
    settings = SimpleNamespace()
    index = create_vector_index(settings)
    assert isinstance(index, InMemoryVectorIndex)


def test_create_vector_index_selects_inmemory_backend() -> None:
    settings = SimpleNamespace(vector_index_backend="inmemory")
    index = create_vector_index(settings)
    assert isinstance(index, InMemoryVectorIndex)


def test_create_vector_index_selects_faiss_backend() -> None:
    settings = SimpleNamespace(vector_index_backend="faiss")
    index = create_vector_index(settings)
    assert isinstance(index, FaissVectorIndex)


def test_create_vector_index_falls_back_for_unknown_backend() -> None:
    settings = SimpleNamespace(vector_index_backend="unknown")
    index = create_vector_index(settings)
    assert isinstance(index, InMemoryVectorIndex)

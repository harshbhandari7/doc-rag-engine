"""Tests for DenseRetriever in src/retrieval/dense.py.

Mocks out Embedder and VectorStore so these run without any model downloads
or network connections.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from src.retrieval.dense import DenseRetriever
from src.vectorstore.store import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_embedder(dim: int = 384) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_dense.return_value = np.zeros((1, dim), dtype=np.float32)
    return embedder


def _mock_store(hits: list[SearchResult] | None = None) -> MagicMock:
    store = MagicMock()
    store.search_dense.return_value = hits or []
    return store


def _hit(text: str = "chunk text", score: float = 0.85, filename: str = "doc.pdf", idx: int = 0) -> SearchResult:
    return SearchResult(
        text=text,
        score=score,
        chunk_strategy="recursive",
        chunk_index=idx,
        chunk_size=len(text),
        parent_source=filename,
        filename=filename,
        format="pdf",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieve_calls_embed_dense_with_query():
    embedder = _mock_embedder()
    retriever = DenseRetriever(embedder, _mock_store())
    retriever.retrieve("what is attention?", collection="rag_recursive", limit=5)
    embedder.embed_dense.assert_called_once_with(["what is attention?"])


def test_retrieve_passes_vector_to_search_dense():
    vec = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    embedder = MagicMock()
    embedder.embed_dense.return_value = vec
    store = _mock_store()
    retriever = DenseRetriever(embedder, store)
    retriever.retrieve("query", collection="rag_test", limit=10)
    # search_dense should receive the vector as a list (via .tolist())
    called_vec = store.search_dense.call_args[0][0]
    assert called_vec == vec[0].tolist()


def test_retrieve_passes_collection_and_limit():
    embedder = _mock_embedder()
    store = _mock_store()
    retriever = DenseRetriever(embedder, store)
    retriever.retrieve("query", collection="rag_fixed", limit=7)
    store.search_dense.assert_called_once()
    _, kwargs = store.search_dense.call_args
    assert kwargs.get("limit") == 7 or store.search_dense.call_args[0][2] == 7


def test_retrieve_passes_filters():
    embedder = _mock_embedder()
    store = _mock_store()
    retriever = DenseRetriever(embedder, store)
    filt = {"source_file": "paper.pdf"}
    retriever.retrieve("query", collection="col", limit=5, filters=filt)
    _, kwargs = store.search_dense.call_args
    assert kwargs.get("filters") == filt or store.search_dense.call_args[0][-1] == filt


def test_retrieve_returns_retrieval_results():
    from src.models import RetrievalResult
    embedder = _mock_embedder()
    store = _mock_store([_hit("some text", score=0.9)])
    retriever = DenseRetriever(embedder, store)
    results = retriever.retrieve("query", "col", limit=5)
    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)


def test_retrieve_method_tagged_dense():
    embedder = _mock_embedder()
    store = _mock_store([_hit()])
    retriever = DenseRetriever(embedder, store)
    results = retriever.retrieve("q", "col", limit=5)
    assert all(r.retrieval_method == "dense" for r in results)


def test_retrieve_score_matches_search_result():
    embedder = _mock_embedder()
    store = _mock_store([_hit(score=0.731)])
    retriever = DenseRetriever(embedder, store)
    results = retriever.retrieve("q", "col", limit=5)
    assert results[0].score == 0.731


def test_retrieve_metadata_mapped_from_search_result():
    embedder = _mock_embedder()
    store = _mock_store([_hit(text="hello", filename="paper.pdf", idx=4)])
    retriever = DenseRetriever(embedder, store)
    results = retriever.retrieve("q", "col", limit=5)
    meta = results[0].metadata
    assert meta["filename"] == "paper.pdf"
    assert meta["chunk_index"] == 4
    assert meta["chunk_strategy"] == "recursive"


def test_retrieve_empty_store_returns_empty():
    embedder = _mock_embedder()
    store = _mock_store([])
    retriever = DenseRetriever(embedder, store)
    results = retriever.retrieve("q", "col", limit=5)
    assert results == []


def test_retrieve_multiple_hits_all_returned():
    hits = [_hit(f"text {i}", score=1.0 - i * 0.1, idx=i) for i in range(5)]
    embedder = _mock_embedder()
    store = _mock_store(hits)
    retriever = DenseRetriever(embedder, store)
    results = retriever.retrieve("q", "col", limit=10)
    assert len(results) == 5

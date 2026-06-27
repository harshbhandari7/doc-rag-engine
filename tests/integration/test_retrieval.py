"""Integration tests for the full retrieval pipeline against live Qdrant.

Requires:
  - docker compose up -d  (Qdrant running on localhost:6333)
  - Collections already populated (python -m src.indexing.index_documents)

Run with:
    pytest tests/integration/ -m integration -v
"""
from __future__ import annotations

import pytest

COLLECTION = "rag_recursive"
QUERY = "How does multi-head attention work?"


@pytest.mark.integration
def test_dense_retriever_returns_results(embedder, store):
    from src.retrieval.dense import DenseRetriever
    retriever = DenseRetriever(embedder, store)
    results = retriever.retrieve(QUERY, COLLECTION, limit=5)
    assert len(results) > 0
    assert all(r.retrieval_method == "dense" for r in results)
    assert all(r.score > 0 for r in results)
    assert all(r.chunk_text.strip() != "" for r in results)


@pytest.mark.integration
def test_bm25_retriever_returns_results(store):
    from src.retrieval.sparse import BM25Retriever
    retriever = BM25Retriever(store, COLLECTION)
    results = retriever.retrieve(QUERY, limit=5)
    assert len(results) > 0
    assert all(r.retrieval_method == "bm25" for r in results)
    assert all(r.score > 0 for r in results)


@pytest.mark.integration
def test_hybrid_rrf_returns_results(embedder, store, cfg):
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.sparse import BM25Retriever
    dense  = DenseRetriever(embedder, store)
    bm25   = BM25Retriever(store, COLLECTION)
    hybrid = HybridRetriever(dense, bm25, cfg)
    results = hybrid.retrieve_rrf(QUERY, COLLECTION, limit=20)
    assert len(results) > 0
    assert all(r.retrieval_method == "hybrid_rrf" for r in results)
    # RRF scores are in (0, 1]
    assert all(0 < r.score <= 1.0 + 1e-9 for r in results)


@pytest.mark.integration
def test_reranker_reorders_candidates(embedder, store, cfg):
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.reranker import Reranker
    from src.retrieval.sparse import BM25Retriever
    dense    = DenseRetriever(embedder, store)
    bm25     = BM25Retriever(store, COLLECTION)
    hybrid   = HybridRetriever(dense, bm25, cfg)
    reranker = Reranker(cfg)

    candidates = hybrid.retrieve_rrf(QUERY, COLLECTION, limit=20)
    top5 = reranker.rerank(QUERY, candidates, top_n=5)

    assert len(top5) == 5
    assert all(r.retrieval_method == "reranked" for r in top5)
    # Reranker scores are cross-encoder relevance probabilities (0–1)
    assert all(0.0 <= r.score <= 1.0 for r in top5)


@pytest.mark.integration
def test_hybrid_rrf_returns_fewer_than_limit_when_sparse(embedder, store, cfg):
    """Limit is a ceiling, not a guarantee — sparse may return fewer candidates."""
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.sparse import BM25Retriever
    dense  = DenseRetriever(embedder, store)
    bm25   = BM25Retriever(store, COLLECTION)
    hybrid = HybridRetriever(dense, bm25, cfg)
    results = hybrid.retrieve_rrf(QUERY, COLLECTION, limit=20)
    assert len(results) <= 20


@pytest.mark.integration
def test_retrieval_metadata_fields_present(embedder, store):
    from src.retrieval.dense import DenseRetriever
    retriever = DenseRetriever(embedder, store)
    results = retriever.retrieve(QUERY, COLLECTION, limit=3)
    for r in results:
        assert "filename" in r.metadata
        assert "chunk_index" in r.metadata
        assert "chunk_strategy" in r.metadata
        assert "chunk_size" in r.metadata

"""Tests for BM25Retriever in src/retrieval/sparse.py.

BM25Retriever builds its index from VectorStore.scroll_all() at construction.
We mock scroll_all so these tests run without Qdrant.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.sparse import BM25Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(payloads: list[dict]) -> MagicMock:
    store = MagicMock()
    store.scroll_all.return_value = payloads
    return store


def _payload(text: str, filename: str = "doc.pdf", chunk_index: int = 0) -> dict:
    return {
        "chunk_text":     text,
        "chunk_strategy": "recursive",
        "chunk_index":    chunk_index,
        "chunk_size":     len(text),
        "parent_source":  filename,
        "filename":       filename,
        "format":         "pdf",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bm25_retriever_builds_index_from_scroll():
    store = _make_store([_payload("hello world"), _payload("another doc")])
    BM25Retriever(store, "rag_test")
    store.scroll_all.assert_called_once_with("rag_test")


def test_retrieve_returns_results_for_matching_query():
    # BM25Okapi IDF = log(N-df+0.5) - log(df+0.5); requires N≥3 for a term
    # appearing in 1 doc to get a positive IDF (at N=2 with df=1 it evaluates to 0).
    store = _make_store([
        _payload("the attention mechanism uses query key value", "attention.pdf", 0),
        _payload("gradient descent optimises the loss function", "optim.pdf", 0),
        _payload("the transformer architecture introduced novel concepts", "transformer.pdf", 0),
    ])
    retriever = BM25Retriever(store, "rag_test")
    results = retriever.retrieve("attention mechanism", limit=10)
    assert len(results) >= 1
    assert any("attention" in r.chunk_text for r in results)


def test_retrieve_returns_results_sorted_by_score_descending():
    store = _make_store([
        _payload("apple banana cherry apple apple"),
        _payload("completely unrelated text about xyz"),
        _payload("apple fruit is nutritious"),
    ])
    retriever = BM25Retriever(store, "rag_test")
    results = retriever.retrieve("apple", limit=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_excludes_zero_score_results():
    store = _make_store([
        _payload("relevant content matching the query"),
        _payload("zzzzz yyyyy xxxxx"),  # no overlap with any normal query
    ])
    retriever = BM25Retriever(store, "rag_test")
    results = retriever.retrieve("relevant content", limit=10)
    assert all(r.score > 0 for r in results)


def test_retrieve_limit_respected():
    payloads = [_payload(f"doc {i} mentions keyword", f"f{i}.pdf", i) for i in range(20)]
    store = _make_store(payloads)
    retriever = BM25Retriever(store, "rag_test")
    results = retriever.retrieve("keyword", limit=5)
    assert len(results) <= 5


def test_retrieve_method_tagged_bm25():
    store = _make_store([_payload("some text with terms")])
    retriever = BM25Retriever(store, "rag_test")
    results = retriever.retrieve("text terms", limit=10)
    assert all(r.retrieval_method == "bm25" for r in results)


def test_retrieve_no_matching_terms_returns_empty():
    store = _make_store([_payload("apple banana cherry")])
    retriever = BM25Retriever(store, "rag_test")
    # Query terms share no overlap with the corpus
    results = retriever.retrieve("zzz yyy xxx", limit=10)
    assert results == []


def test_retrieve_metadata_mapped_correctly():
    # Need N≥3 docs for unique-term IDF to be positive (see test_retrieve_returns_results comment)
    store = _make_store([
        _payload("the query term appears here", "paper.pdf", 3),
        _payload("completely different material", "other1.pdf", 0),
        _payload("another unrelated document", "other2.pdf", 0),
    ])
    retriever = BM25Retriever(store, "rag_test")
    results = retriever.retrieve("query term", limit=1)
    assert len(results) == 1
    assert results[0].metadata["filename"] == "paper.pdf"
    assert results[0].metadata["chunk_index"] == 3
    assert results[0].metadata["chunk_strategy"] == "recursive"


def test_retrieve_chunk_text_preserved():
    text = "verbatim preservation check"
    store = _make_store([
        _payload(text),
        _payload("something completely unrelated"),
        _payload("another unrelated passage"),
    ])
    retriever = BM25Retriever(store, "rag_test")
    results = retriever.retrieve("verbatim preservation", limit=1)
    assert len(results) == 1
    assert results[0].chunk_text == text

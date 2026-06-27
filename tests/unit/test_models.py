"""Tests for the RetrievalResult dataclass in src/models.py."""
from __future__ import annotations

from src.models import RetrievalResult


def test_retrieval_result_fields():
    r = RetrievalResult(
        chunk_text="hello",
        score=0.85,
        metadata={"filename": "doc.pdf", "chunk_index": 3},
        retrieval_method="dense",
    )
    assert r.chunk_text == "hello"
    assert r.score == 0.85
    assert r.retrieval_method == "dense"


def test_retrieval_result_metadata_access():
    r = RetrievalResult(
        chunk_text="text",
        score=0.5,
        metadata={"filename": "report.pdf", "chunk_index": 7, "chunk_strategy": "recursive"},
        retrieval_method="bm25",
    )
    assert r.metadata["filename"] == "report.pdf"
    assert r.metadata["chunk_index"] == 7
    assert r.metadata.get("missing_key") is None


def test_retrieval_result_zero_score():
    r = RetrievalResult(chunk_text="x", score=0.0, metadata={}, retrieval_method="bm25")
    assert r.score == 0.0


def test_retrieval_result_equality():
    r1 = RetrievalResult(chunk_text="x", score=0.5, metadata={"k": 1}, retrieval_method="dense")
    r2 = RetrievalResult(chunk_text="x", score=0.5, metadata={"k": 1}, retrieval_method="dense")
    assert r1 == r2


def test_retrieval_result_inequality_on_score():
    r1 = RetrievalResult(chunk_text="x", score=0.5, metadata={}, retrieval_method="dense")
    r2 = RetrievalResult(chunk_text="x", score=0.9, metadata={}, retrieval_method="dense")
    assert r1 != r2

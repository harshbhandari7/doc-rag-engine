"""Tests for chunking logic in src/chunking/chunkers.py.

Covers:
  - FixedSizeChunker and RecursiveChunker with fake LoadedDocuments
  - Private helpers: _split_sentences, _cosine_distances, _make_chunks
"""
from __future__ import annotations

import numpy as np
import pytest

from src.chunking.chunkers import (
    ChunkedDocument,
    FixedSizeChunker,
    RecursiveChunker,
    _cosine_distances,
    _make_chunks,
    _split_sentences,
)
from tests.conftest import make_doc, make_metadata


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    sentences = _split_sentences("Hello world. How are you? I am fine!")
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you?"
    assert sentences[2] == "I am fine!"


def test_split_sentences_single():
    sentences = _split_sentences("Only one sentence here.")
    assert sentences == ["Only one sentence here."]


def test_split_sentences_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


def test_split_sentences_strips_whitespace():
    sentences = _split_sentences("  First sentence.  Second sentence.  ")
    assert all(s == s.strip() for s in sentences)


# ---------------------------------------------------------------------------
# _cosine_distances
# ---------------------------------------------------------------------------

def test_cosine_distances_shape():
    vecs = np.random.rand(5, 16).astype(np.float32)
    dists = _cosine_distances(vecs)
    assert dists.shape == (4,)


def test_cosine_distances_identical_vectors():
    v = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    dists = _cosine_distances(v)
    assert len(dists) == 1
    assert abs(dists[0]) < 1e-6  # identical → distance 0


def test_cosine_distances_orthogonal_vectors():
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dists = _cosine_distances(v)
    assert abs(dists[0] - 1.0) < 1e-6  # orthogonal → distance 1


def test_cosine_distances_range():
    vecs = np.random.rand(10, 32).astype(np.float32)
    dists = _cosine_distances(vecs)
    assert np.all(dists >= -1e-6)   # distance ≥ 0
    assert np.all(dists <= 2.0 + 1e-6)  # distance ≤ 2


# ---------------------------------------------------------------------------
# _make_chunks
# ---------------------------------------------------------------------------

def test_make_chunks_basic():
    meta = make_metadata("test.pdf", "pdf")
    chunks = _make_chunks(["first chunk", "second chunk"], meta, "fixed")
    assert len(chunks) == 2
    assert chunks[0].text == "first chunk"
    assert chunks[0].chunk_strategy == "fixed"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_make_chunks_filters_whitespace_only():
    meta = make_metadata()
    chunks = _make_chunks(["valid", "   ", "\n\n", "also valid"], meta, "recursive")
    assert len(chunks) == 2
    assert chunks[0].text == "valid"
    assert chunks[1].text == "also valid"


def test_make_chunks_sets_parent_source():
    meta = make_metadata("report.pdf")
    chunks = _make_chunks(["text"], meta, "semantic")
    assert chunks[0].parent_source == "report.pdf"


def test_make_chunks_chunk_size_is_len():
    meta = make_metadata()
    text = "exactly twenty chars"
    chunks = _make_chunks([text], meta, "fixed")
    assert chunks[0].chunk_size == len(text)


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

def test_fixed_chunker_produces_chunks():
    doc = make_doc("a" * 600)  # longer than default chunk_size=512
    chunks = FixedSizeChunker().chunk([doc])
    assert len(chunks) >= 1


def test_fixed_chunker_respects_chunk_size():
    from configs.settings import AppConfig
    cfg = AppConfig()
    doc = make_doc("x" * 2000)
    chunks = FixedSizeChunker(cfg).chunk([doc])
    # All chunks must be at most chunk_size + overlap characters
    for c in chunks:
        assert c.chunk_size <= cfg.chunk_size + cfg.chunk_overlap + 1


def test_fixed_chunker_strategy_label():
    doc = make_doc("short text")
    chunks = FixedSizeChunker().chunk([doc])
    for c in chunks:
        assert c.chunk_strategy == "fixed"


def test_fixed_chunker_returns_chunked_documents():
    doc = make_doc("hello world")
    chunks = FixedSizeChunker().chunk([doc])
    assert all(isinstance(c, ChunkedDocument) for c in chunks)


def test_fixed_chunker_multiple_docs():
    docs = [make_doc("doc one content", "a.txt"), make_doc("doc two content", "b.txt")]
    chunks = FixedSizeChunker().chunk(docs)
    sources = {c.parent_source for c in chunks}
    assert "a.txt" in sources
    assert "b.txt" in sources


def test_fixed_chunker_empty_page_produces_no_chunks():
    doc = make_doc("   ")  # whitespace only
    chunks = FixedSizeChunker().chunk([doc])
    assert len(chunks) == 0


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

def test_recursive_chunker_strategy_label():
    doc = make_doc("paragraph one.\n\nparagraph two.")
    chunks = RecursiveChunker().chunk([doc])
    for c in chunks:
        assert c.chunk_strategy == "recursive"


def test_recursive_chunker_respects_paragraph_boundary():
    # Each segment must exceed chunk_size (512) to force a split at \n\n.
    # At 300 chars each the total is 602 > 512, so the splitter tries \n\n first.
    text = "A" * 300 + "\n\n" + "B" * 300
    doc = make_doc(text)
    chunks = RecursiveChunker().chunk([doc])
    texts = [c.text.lstrip() for c in chunks]
    assert any(t.startswith("A") for t in texts)
    assert any(t.startswith("B") for t in texts)


def test_recursive_chunker_chunk_indices_sequential():
    doc = make_doc("x" * 2000)
    chunks = RecursiveChunker().chunk([doc])
    for expected_idx, c in enumerate(chunks):
        assert c.chunk_index == expected_idx


def test_recursive_chunker_preserves_metadata():
    meta = make_metadata("my_doc.pdf", "pdf")
    from src.ingestion.models import LoadedDocument, PageContent, LoadStatus
    doc = LoadedDocument(
        content="some text",
        pages=[PageContent(page_number=1, text="some text")],
        metadata=meta,
        status=LoadStatus.SUCCESS,
    )
    chunks = RecursiveChunker().chunk([doc])
    for c in chunks:
        assert c.metadata.filename == "my_doc.pdf"
        assert c.metadata.format == "pdf"

"""Shared pytest fixtures used across unit and integration tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.models import DocumentMetadata, LoadStatus, LoadedDocument, PageContent
from src.models import RetrievalResult
from src.vectorstore.store import SearchResult


# ---------------------------------------------------------------------------
# Document / chunk factories
# ---------------------------------------------------------------------------

def make_metadata(filename: str = "test.txt", fmt: str = "txt") -> DocumentMetadata:
    return DocumentMetadata(
        source=Path(f"data/raw/{filename}"),
        filename=filename,
        format=fmt,
        file_size_bytes=1000,
        total_pages=1,
        loaded_pages=1,
    )


def make_doc(text: str = "Hello world. This is a test.", filename: str = "test.txt") -> LoadedDocument:
    meta = make_metadata(filename)
    return LoadedDocument(
        content=text,
        pages=[PageContent(page_number=1, text=text)],
        metadata=meta,
        status=LoadStatus.SUCCESS,
    )


def make_result(
    chunk_text: str = "some chunk text",
    score: float = 0.9,
    filename: str = "doc.pdf",
    chunk_index: int = 0,
    method: str = "dense",
    chunk_strategy: str = "recursive",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_text=chunk_text,
        score=score,
        metadata={
            "filename":       filename,
            "chunk_index":    chunk_index,
            "chunk_strategy": chunk_strategy,
            "chunk_size":     len(chunk_text),
            "parent_source":  filename,
            "format":         "pdf",
        },
        retrieval_method=method,
    )


def make_search_result(
    text: str = "some chunk text",
    score: float = 0.9,
    filename: str = "doc.pdf",
    chunk_index: int = 0,
) -> SearchResult:
    return SearchResult(
        text=text,
        score=score,
        chunk_strategy="recursive",
        chunk_index=chunk_index,
        chunk_size=len(text),
        parent_source=filename,
        filename=filename,
        format="pdf",
    )


# ---------------------------------------------------------------------------
# Fixtures (used via dependency injection in test functions)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_doc():
    return make_doc()


@pytest.fixture
def sample_result():
    return make_result()


@pytest.fixture
def two_results():
    """Two distinct RetrievalResults for fusion tests."""
    return [
        make_result(chunk_text="alpha chunk", filename="a.pdf", chunk_index=0, score=0.9),
        make_result(chunk_text="beta chunk",  filename="b.pdf", chunk_index=0, score=0.8),
    ]

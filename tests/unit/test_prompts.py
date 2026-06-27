"""Tests for format_context in src/generation/prompts.py."""
from __future__ import annotations

from src.generation.prompts import format_context
from src.models import RetrievalResult


def _r(text: str, filename: str = "doc.pdf", chunk_index: int = 0) -> RetrievalResult:
    return RetrievalResult(
        chunk_text=text,
        score=0.9,
        metadata={"filename": filename, "chunk_index": chunk_index},
        retrieval_method="dense",
    )


def test_empty_list_returns_empty_string():
    assert format_context([]) == ""


def test_single_result_numbered():
    ctx = format_context([_r("The attention mechanism allows…", "paper.pdf", 14)])
    assert "[1]" in ctx
    assert "Source: paper.pdf" in ctx
    assert "Chunk 14" in ctx
    assert "The attention mechanism allows" in ctx


def test_two_results_numbered_sequentially():
    ctx = format_context([
        _r("first passage", "a.pdf", 0),
        _r("second passage", "b.pdf", 5),
    ])
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "a.pdf" in ctx
    assert "b.pdf" in ctx


def test_results_separated_by_blank_line():
    ctx = format_context([_r("one", "a.pdf", 0), _r("two", "b.pdf", 1)])
    # Each chunk is separated by a double newline
    assert "\n\n" in ctx


def test_chunk_text_is_stripped():
    ctx = format_context([_r("  text with whitespace  ", "f.pdf", 0)])
    assert "text with whitespace" in ctx
    # Leading/trailing spaces on the chunk text should be gone
    lines = ctx.split("\n")
    chunk_line = lines[1]  # second line is the chunk text
    assert chunk_line == chunk_line.strip()


def test_missing_filename_falls_back_to_unknown():
    r = RetrievalResult(
        chunk_text="content",
        score=0.5,
        metadata={"chunk_index": 0},  # no filename key
        retrieval_method="dense",
    )
    ctx = format_context([r])
    assert "Source: unknown" in ctx


def test_format_preserves_order():
    results = [_r(f"passage {i}", f"doc{i}.pdf", i) for i in range(5)]
    ctx = format_context(results)
    for i in range(1, 6):
        assert f"[{i}]" in ctx
    # Verify order: [1] appears before [5]
    assert ctx.index("[1]") < ctx.index("[5]")

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from configs.settings import AppConfig
from src.embedding.embedder import Embedder
from src.ingestion.models import DocumentMetadata, LoadedDocument


@dataclass
class ChunkedDocument:
    """A single chunk produced by any chunking strategy.

    Carries the full original :class:`~src.ingestion.models.DocumentMetadata`
    forward plus the chunk-specific fields below.  Downstream code (vector
    store, retrieval, evaluation) should import this type from here; if it
    ever needs to live without the chunker logic it can be moved to a shared
    models module.

    Fields:
        text:             The chunk text.
        metadata:         Unchanged metadata from the source document.
        chunk_strategy:   One of ``"fixed"``, ``"recursive"``, ``"semantic"``.
        chunk_index:      0-based position among chunks produced from this document.
        chunk_size:       Actual character count of *text* (``len(text)``).
        parent_source:    ``metadata.filename`` of the originating document.
    """

    text: str
    metadata: DocumentMetadata
    chunk_strategy: str
    chunk_index: int
    chunk_size: int
    parent_source: str


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, docs: list[LoadedDocument]) -> list[ChunkedDocument]:
        """Split *docs* and return a flat list of :class:`ChunkedDocument` objects."""


class FixedSizeChunker(BaseChunker):
    """Splits each page by character count with no separator awareness.

    Uses ``separator=""`` so cuts may fall mid-sentence or mid-word.
    Useful as a lower-bound baseline. ``chunk_size`` and ``chunk_overlap``
    are read from :class:`~configs.settings.AppConfig` (env: ``APP_CHUNK_SIZE``,
    ``APP_CHUNK_OVERLAP``).
    """

    STRATEGY = "fixed"

    def __init__(self, config: AppConfig | None = None) -> None:
        cfg = config or AppConfig()
        self._splitter = CharacterTextSplitter(
            separator="",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            strip_whitespace=False,
        )

    def chunk(self, docs: list[LoadedDocument]) -> list[ChunkedDocument]:
        results: list[ChunkedDocument] = []
        for doc in docs:
            texts: list[str] = []
            for page in doc.pages:
                texts.extend(self._splitter.split_text(page.text))
            results.extend(_make_chunks(texts, doc.metadata, self.STRATEGY))
        return results


class RecursiveChunker(BaseChunker):
    """Splits each page using a hierarchy of separators.

    Tries ``\\n\\n`` → ``\\n`` → ``". "`` → ``"! "`` → ``"? "`` → ``" "`` →
    ``""`` in order, falling back to the next separator when a piece still
    exceeds ``chunk_size``.  Respects paragraph and sentence boundaries where
    possible while guaranteeing the size limit.
    """

    STRATEGY = "recursive"

    def __init__(self, config: AppConfig | None = None) -> None:
        cfg = config or AppConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def chunk(self, docs: list[LoadedDocument]) -> list[ChunkedDocument]:
        results: list[ChunkedDocument] = []
        for doc in docs:
            texts: list[str] = []
            for page in doc.pages:
                texts.extend(self._splitter.split_text(page.text))
            results.extend(_make_chunks(texts, doc.metadata, self.STRATEGY))
        return results


class SemanticChunker(BaseChunker):
    """Splits by detecting semantic breaks between sentences via embedding cosine distance.

    Algorithm:
    1. Sentence-tokenise the text.
    2. Embed all sentences with the provided :class:`~src.embedder.Embedder`.
    3. Compute cosine distance between each consecutive sentence pair.
    4. Any transition whose distance exceeds the *breakpoint_percentile*-th
       percentile of all distances becomes a chunk boundary.
    5. Sentences between boundaries are joined into one chunk string.

    **Why full-document text instead of page-by-page:**
    Page boundaries are a physical artefact of the PDF layout, not semantic
    breaks.  A paragraph that straddles pages 4 and 5 would generate a
    spurious high cosine distance at that page join if we chunked per-page.
    Concatenating all pages first lets the distance signal reflect true topic
    shifts.  Fixed and recursive chunkers do not need this because they operate
    on characters, not meaning.

    **Note on LangChain's SemanticChunker:**
    ``langchain_experimental.text_splitter.SemanticChunker`` implements the
    same percentile-distance algorithm but requires a LangChain
    ``Embeddings`` object (``embed_documents`` / ``embed_query`` interface).
    A two-line adapter would suffice if you ever want to use it::

        from langchain_core.embeddings import Embeddings

        class _Adapter(Embeddings):
            def __init__(self, e: Embedder): self._e = e
            def embed_documents(self, texts):  return self._e.embed_dense(texts).tolist()
            def embed_query(self, text):        return self._e.embed_dense([text])[0].tolist()

    ``langchain-experimental`` is intentionally not added as a dependency
    here to avoid the large transitive install.
    """

    STRATEGY = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        config: AppConfig | None = None,
        breakpoint_percentile: int | None = None,
    ) -> None:
        cfg = config or AppConfig()
        self._embedder = embedder
        # explicit arg wins over config so callers can override per-call
        self._percentile = breakpoint_percentile if breakpoint_percentile is not None else cfg.semantic_breakpoint_threshold
        self._max_chunk_size = cfg.chunk_size
        # Fallback splitter applied only to chunks that exceed the size ceiling.
        self._overflow_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def chunk(self, docs: list[LoadedDocument]) -> list[ChunkedDocument]:
        results: list[ChunkedDocument] = []
        for doc in docs:
            # Concatenate all pages — see class docstring for why.
            full_text = "\n\n".join(page.text for page in doc.pages)
            texts = self._cap_oversized(self._split(full_text))
            results.extend(_make_chunks(texts, doc.metadata, self.STRATEGY))
        return results

    def _cap_oversized(self, chunks: list[str]) -> list[str]:
        """Re-split any chunk that exceeds the character ceiling using the recursive splitter."""
        result: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self._max_chunk_size:
                result.append(chunk)
            else:
                result.extend(self._overflow_splitter.split_text(chunk))
        return result

    def _split(self, text: str) -> list[str]:
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return sentences

        vectors = self._embedder.embed_dense(sentences)
        distances = _cosine_distances(vectors)
        threshold = float(np.percentile(distances, self._percentile))

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        for i, dist in enumerate(distances):
            if dist > threshold:
                chunks.append(" ".join(current))
                current = []
            current.append(sentences[i + 1])
        if current:
            chunks.append(" ".join(current))
        return chunks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_chunks(
    texts: list[str],
    metadata: DocumentMetadata,
    strategy: str,
) -> list[ChunkedDocument]:
    return [
        ChunkedDocument(
            text=text,
            metadata=metadata,
            chunk_strategy=strategy,
            chunk_index=i,
            chunk_size=len(text),
            parent_source=metadata.filename,
        )
        for i, text in enumerate(texts)
        if text.strip()
    ]


def _split_sentences(text: str) -> list[str]:
    """Split *text* on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]


def _cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Cosine distance between each consecutive row pair in *vectors*.

    Returns an array of shape ``(len(vectors) - 1,)``.
    Range 0 (identical) to 2 (opposite); typical breakpoint thresholds fall
    in 0.2–0.6 for sentence-transformer embeddings.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / norms
    similarities = np.einsum("ij,ij->i", normalized[:-1], normalized[1:])
    return 1.0 - similarities

from __future__ import annotations

import logging

from src.embedding.embedder import Embedder
from src.models import RetrievalResult
from src.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Retrieve chunks by semantic similarity using dense vector search.

    Embeds the query with a bi-encoder (MiniLM) and finds the nearest
    neighbours in the collection using cosine similarity.  Strength: matches
    meaning across paraphrases ("how does attention work" → "mechanism of
    self-attention").  Weakness: misses exact keyword matches when the semantic
    neighbourhood is too broad ("BLEU score" → generic evaluation chunks).

    The ``Embedder`` and ``VectorStore`` are injected rather than instantiated
    here so the calling pipeline can share one instance across dense, sparse,
    and hybrid retrievers — loading the embedding model three times would be
    wasteful.

    Usage::

        retriever = DenseRetriever(embedder, store)
        results = retriever.retrieve(
            query="How does attention work?",
            collection="rag_recursive",
            limit=20,
        )
    """

    METHOD = "dense"

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        collection: str,
        limit: int = 20,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Embed *query* and return the top-*limit* semantically similar chunks.

        Args:
            query:      User query string.
            collection: Qdrant collection to search (e.g. ``"rag_recursive"``).
            limit:      Number of results to return.  Defaults to 20 — a broad
                        candidate set intended for downstream reranking.
            filters:    Optional metadata filter, e.g. ``{"source_file": "paper.pdf"}``.
                        Converted to a Qdrant Filter inside VectorStore.

        Returns:
            List of :class:`~src.models.RetrievalResult` sorted by descending
            cosine similarity score.
        """
        logger.debug("Dense retrieve  query=%r  collection=%s  limit=%d", query, collection, limit)

        query_vector = self._embedder.embed_dense([query])[0].tolist()
        hits = self._store.search_dense(query_vector, collection, limit=limit, filters=filters)

        return [_to_result(h) for h in hits]


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _to_result(hit) -> RetrievalResult:
    return RetrievalResult(
        chunk_text=hit.text,
        score=hit.score,
        metadata={
            "chunk_strategy": hit.chunk_strategy,
            "chunk_index":    hit.chunk_index,
            "chunk_size":     hit.chunk_size,
            "parent_source":  hit.parent_source,
            "filename":       hit.filename,
            "format":         hit.format,
        },
        retrieval_method=DenseRetriever.METHOD,
    )

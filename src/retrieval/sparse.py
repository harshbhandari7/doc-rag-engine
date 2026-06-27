from __future__ import annotations

import logging

import numpy as np
from rank_bm25 import BM25Okapi

from src.embedding.embedder import Embedder
from src.models import RetrievalResult
from src.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Classical BM25 retrieval over an in-memory index built from a Qdrant collection.

    BM25 (Okapi BM25) is a statistical ranking function based on term frequency
    and inverse document frequency.  No neural network — just word overlap and
    corpus statistics.  Strength: exact keyword matches, fast, fully
    interpretable (you can see which terms drove the score).  Weakness: no
    semantic understanding ("car" does not match "automobile").

    The index is built once at construction by scrolling all chunk texts from
    Qdrant.  For this dataset (~10–16k chunks per collection) this is fine in
    memory.  At millions of chunks, replace this with a dedicated search engine
    such as Elasticsearch or OpenSearch.

    Comparing BM25 against SPLADE (see :class:`SparseRetriever`) on the same
    query is informative: gaps in recall show you exactly where learned term
    expansion helps.

    Usage::

        retriever = BM25Retriever(store, collection="rag_recursive")
        results = retriever.retrieve("BLEU score evaluation", limit=20)
    """

    METHOD = "bm25"

    def __init__(self, store: VectorStore, collection: str) -> None:
        logger.info("Building BM25 index from %s...", collection)
        payloads = store.scroll_all(collection)

        self._texts: list[str] = [p.get("chunk_text", "") for p in payloads]
        self._metadatas: list[dict] = [
            {
                "chunk_strategy": p.get("chunk_strategy", ""),
                "chunk_index":    p.get("chunk_index", 0),
                "chunk_size":     p.get("chunk_size", 0),
                "parent_source":  p.get("parent_source", ""),
                "filename":       p.get("filename", ""),
                "format":         p.get("format", ""),
            }
            for p in payloads
        ]

        tokenized_corpus = [text.lower().split() for text in self._texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index ready — %d documents", len(self._texts))

    def retrieve(self, query: str, limit: int = 20) -> list[RetrievalResult]:
        """Score all indexed chunks against *query* and return the top *limit*.

        Args:
            query: User query string.  Tokenised by whitespace; lowercased.
            limit: Number of results.  Zero-score chunks (no query term overlap)
                   are excluded before applying the limit.

        Returns:
            List of :class:`~src.models.RetrievalResult` sorted by descending
            BM25 score, tagged ``retrieval_method="bm25"``.
        """
        logger.debug("BM25 retrieve  query=%r  limit=%d", query, limit)
        tokens = query.lower().split()
        scores: np.ndarray = self._bm25.get_scores(tokens)

        # Exclude zero-score chunks — they share no terms with the query.
        top_indices = np.argsort(scores)[::-1]
        top_indices = [i for i in top_indices if scores[i] > 0][:limit]

        return [
            RetrievalResult(
                chunk_text=self._texts[i],
                score=float(scores[i]),
                metadata=self._metadatas[i],
                retrieval_method=self.METHOD,
            )
            for i in top_indices
        ]


class SparseRetriever:
    """SPLADE sparse retrieval via Qdrant's sparse vector index.

    Unlike BM25, SPLADE is a *learned* sparse model — it encodes text into a
    sparse vector over a large vocabulary where related terms share activated
    dimensions.  "Car" and "automobile" will activate overlapping dimensions,
    giving SPLADE recall that BM25 lacks while retaining sparse vector
    efficiency.

    Embeds the query with :meth:`~src.embedding.Embedder.embed_sparse` (SPLADE)
    and queries Qdrant's sparse index directly.  Qdrant computes the dot product
    between the query sparse vector and all indexed sparse vectors.

    Usage::

        retriever = SparseRetriever(embedder, store)
        results = retriever.retrieve(
            query="BLEU score evaluation",
            collection="rag_recursive",
            limit=20,
        )
    """

    METHOD = "sparse"

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        collection: str,
        limit: int = 20,
    ) -> list[RetrievalResult]:
        """Embed *query* with SPLADE and return the top-*limit* matching chunks.

        Args:
            query:      User query string.
            collection: Qdrant collection to search.
            limit:      Number of results to return.

        Returns:
            List of :class:`~src.models.RetrievalResult` sorted by descending
            SPLADE dot-product score, tagged ``retrieval_method="sparse"``.
        """
        logger.debug("SPLADE retrieve  query=%r  collection=%s  limit=%d", query, collection, limit)
        query_sparse = self._embedder.embed_sparse([query])[0]
        hits = self._store.search_sparse(query_sparse, collection, limit=limit)

        return [
            RetrievalResult(
                chunk_text=h.text,
                score=h.score,
                metadata={
                    "chunk_strategy": h.chunk_strategy,
                    "chunk_index":    h.chunk_index,
                    "chunk_size":     h.chunk_size,
                    "parent_source":  h.parent_source,
                    "filename":       h.filename,
                    "format":         h.format,
                },
                retrieval_method=self.METHOD,
            )
            for h in hits
        ]

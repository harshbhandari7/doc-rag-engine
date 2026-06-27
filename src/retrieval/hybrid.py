from __future__ import annotations

import logging
from typing import Union

from configs.settings import AppConfig
from src.models import RetrievalResult
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import BM25Retriever, SparseRetriever

logger = logging.getLogger(__name__)

_AnySparseRetriever = Union[BM25Retriever, SparseRetriever]


class HybridRetriever:
    """Fuse dense and sparse retrieval results into a single ranked list.

    Implements two fusion strategies:

    * :meth:`retrieve_rrf` — Reciprocal Rank Fusion.  Uses only rank position,
      not raw scores, so incompatible score scales (cosine vs BM25) are never
      directly compared.  The manual implementation here can be diffed against
      Qdrant's native RRF (``VectorStore.search_hybrid``) on the evaluation
      dashboard — discrepancies reveal implementation subtleties.

    * :meth:`retrieve_weighted` — Weighted score combination.  Min-max
      normalises each result set to [0, 1], then combines as
      ``alpha * dense + (1 - alpha) * sparse``.  Requires score scales to be
      meaningful within each list; lets you tune the semantic/keyword balance.

    Both methods accept either :class:`BM25Retriever` or :class:`SparseRetriever`
    on the sparse side.  ``BM25Retriever`` is collection-agnostic (the collection
    is fixed at construction); ``SparseRetriever`` takes collection per-call.
    The dispatcher :meth:`_call_sparse` handles both transparently.

    Usage::

        hybrid = HybridRetriever(dense_retriever, bm25_retriever)
        results_rrf      = hybrid.retrieve_rrf("attention mechanism", "rag_recursive", limit=20)
        results_weighted = hybrid.retrieve_weighted("attention mechanism", "rag_recursive", limit=20)
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: _AnySparseRetriever,
        config: AppConfig | None = None,
    ) -> None:
        cfg = config or AppConfig()
        self._dense   = dense
        self._sparse  = sparse
        self._rrf_k   = cfg.rrf_k
        self._alpha   = cfg.hybrid_alpha

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    def retrieve_rrf(
        self,
        query: str,
        collection: str,
        limit: int = 20,
        k: int | None = None,
    ) -> list[RetrievalResult]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion.

        Args:
            query:      User query string.
            collection: Qdrant collection (ignored by BM25Retriever).
            limit:      Number of results to return after fusion.
            k:          RRF damping constant.  Defaults to ``AppConfig.rrf_k``
                        (60).  Lower k amplifies the top ranks; higher k
                        flattens the distribution.

        Returns:
            Fused list tagged ``retrieval_method="hybrid_rrf"``, sorted by
            descending RRF score.
        """
        k = k if k is not None else self._rrf_k

        dense_results  = self._dense.retrieve(query, collection, limit=limit)
        sparse_results = self._call_sparse(query, collection, limit=limit)

        logger.debug(
            "RRF fusion  dense=%d  sparse=%d  k=%d",
            len(dense_results), len(sparse_results), k,
        )
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=k)
        return fused[:limit]

    def retrieve_weighted(
        self,
        query: str,
        collection: str,
        limit: int = 20,
        alpha: float | None = None,
    ) -> list[RetrievalResult]:
        """Fuse dense and sparse results via weighted normalised score combination.

        Scores within each result set are min-max normalised to [0, 1] before
        combining, so incompatible raw score scales do not bias the result.

        Args:
            query:      User query string.
            collection: Qdrant collection (ignored by BM25Retriever).
            limit:      Number of results to return after fusion.
            alpha:      Weight for the dense score (0–1).  Defaults to
                        ``AppConfig.hybrid_alpha`` (0.7).  alpha=1.0 is
                        pure dense; alpha=0.0 is pure sparse.

        Returns:
            Fused list tagged ``retrieval_method="hybrid_weighted"``, sorted
            by descending combined score.
        """
        alpha = alpha if alpha is not None else self._alpha

        dense_results  = self._dense.retrieve(query, collection, limit=limit)
        sparse_results = self._call_sparse(query, collection, limit=limit)

        logger.debug(
            "Weighted fusion  dense=%d  sparse=%d  alpha=%.2f",
            len(dense_results), len(sparse_results), alpha,
        )
        fused = weighted_score_fusion(dense_results, sparse_results, alpha=alpha)
        return fused[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_sparse(self, query: str, collection: str, limit: int) -> list[RetrievalResult]:
        """Dispatch to the correct sparse retriever interface."""
        if isinstance(self._sparse, BM25Retriever):
            # BM25Retriever is initialised with a fixed collection; no per-call collection.
            return self._sparse.retrieve(query, limit=limit)
        return self._sparse.retrieve(query, collection, limit=limit)


# ---------------------------------------------------------------------------
# Pure fusion functions — stateless, easy to test in isolation
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    k: int = 60,
) -> list[RetrievalResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score for a chunk = Σ  1 / (k + rank_i)  across all lists it appears in.
    Rank is 1-based.  Chunks absent from a list contribute 0 from that list.

    Score scales of the two input lists are irrelevant — only rank position matters.
    The k=60 constant from the original paper (Cormack et al., 2009) dampens the
    outsized influence of the very top rank.

    Args:
        dense_results:  Ranked list from dense retrieval.
        sparse_results: Ranked list from sparse retrieval (BM25 or SPLADE).
        k:              Damping constant.

    Returns:
        Merged list sorted by descending RRF score, tagged
        ``retrieval_method="hybrid_rrf"``.
    """
    scores: dict[tuple, float] = {}
    origin: dict[tuple, RetrievalResult] = {}

    for rank, result in enumerate(dense_results, start=1):
        key = _chunk_key(result)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        origin[key] = result

    for rank, result in enumerate(sparse_results, start=1):
        key = _chunk_key(result)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        origin.setdefault(key, result)

    ranked_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        RetrievalResult(
            chunk_text=origin[k].chunk_text,
            score=scores[k],
            metadata=origin[k].metadata,
            retrieval_method="hybrid_rrf",
        )
        for k in ranked_keys
    ]


def weighted_score_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    alpha: float = 0.7,
) -> list[RetrievalResult]:
    """Merge two result sets using min-max normalised weighted score combination.

    Each result set is normalised independently to [0, 1]:
        normalised_score = (score - min) / (max - min)

    If all scores in a list are identical, every normalised score is 1.0.
    Chunks absent from one list receive a normalised score of 0 for that list.

    Combined score = alpha * dense_norm + (1 - alpha) * sparse_norm

    Args:
        dense_results:  Results from dense retrieval.
        sparse_results: Results from sparse retrieval.
        alpha:          Dense weight (0–1).  0.7 = 70% semantic, 30% keyword.

    Returns:
        Merged list sorted by descending combined score, tagged
        ``retrieval_method="hybrid_weighted"``.
    """
    dense_norm  = _normalise(dense_results)
    sparse_norm = _normalise(sparse_results)

    all_keys: set[tuple] = set(dense_norm) | set(sparse_norm)
    origin: dict[tuple, RetrievalResult] = {
        **{_chunk_key(r): r for r in sparse_results},
        **{_chunk_key(r): r for r in dense_results},
    }

    combined: dict[tuple, float] = {
        key: alpha * dense_norm.get(key, 0.0) + (1 - alpha) * sparse_norm.get(key, 0.0)
        for key in all_keys
    }

    ranked_keys = sorted(combined, key=combined.__getitem__, reverse=True)
    return [
        RetrievalResult(
            chunk_text=origin[k].chunk_text,
            score=combined[k],
            metadata=origin[k].metadata,
            retrieval_method="hybrid_weighted",
        )
        for k in ranked_keys
    ]


def _chunk_key(result: RetrievalResult) -> tuple:
    """Unique identifier for a chunk: (filename, chunk_index).

    Using text as a key would fail for duplicate chunks; using the index pair
    is stable and collision-free across the corpus.
    """
    return (result.metadata.get("filename", ""), result.metadata.get("chunk_index", -1))


def _normalise(results: list[RetrievalResult]) -> dict[tuple, float]:
    """Min-max normalise scores to [0, 1], keyed by chunk identity."""
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    denom = (hi - lo) or 1.0  # avoid division by zero when all scores are equal
    return {
        _chunk_key(r): (r.score - lo) / denom
        for r in results
    }

from __future__ import annotations

import logging

from flashrank import RerankRequest
from flashrank import Ranker as FlashRanker

from configs.settings import AppConfig
from src.models import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using FlashRank.

    Vector similarity is a proxy for relevance — it measures whether two pieces
    of text are close in embedding space, not whether one actually answers the
    other.  A cross-encoder is fundamentally different: it reads the query and
    the candidate chunk *together* as a single input and produces a direct
    relevance score.  This joint encoding lets the model see how query terms
    interact with specific passage content, which is far more accurate than
    comparing independent embeddings.

    Cross-encoders are too slow to run over an entire corpus, so this sits after
    a fast approximate retrieval step:

    1. Retrieve a broad candidate set (e.g. top-20) via hybrid search.
    2. Pass candidates to :meth:`rerank` — the cross-encoder scores each one.
    3. Return the top-N reranked results as context to the LLM.

    The default model (``ms-marco-MiniLM-L-12-v2``) is ~90 MB, downloads to
    ``/tmp`` on first use, and was trained specifically for passage ranking on
    MS MARCO.  Override via ``APP_RERANK_MODEL``.

    Usage::

        reranker = Reranker()
        candidates: list[RetrievalResult] = hybrid.retrieve_rrf(query, collection, limit=20)
        top5 = reranker.rerank(query, candidates, top_n=5)
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        cfg = config or AppConfig()
        logger.info("Loading rerank model %s...", cfg.rerank_model)
        self._ranker = FlashRanker(model_name=cfg.rerank_model, max_length=512)
        logger.info("Rerank model ready.")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank *results* for *query* using a cross-encoder.

        FlashRank scores each (query, passage) pair jointly, then returns them
        sorted by relevance.  The original metadata from each ``RetrievalResult``
        is preserved — FlashRank only contributes the new score.

        Args:
            query:   The original user query string.
            results: Candidate ``RetrievalResult`` objects from hybrid retrieval
                     (typically top-20).
            top_n:   How many results to return after reranking.  ``None``
                     returns all candidates in reranked order.

        Returns:
            ``RetrievalResult`` objects sorted by descending cross-encoder score,
            truncated to *top_n*.  Each result's ``score`` is replaced with the
            cross-encoder relevance probability and ``retrieval_method`` is set
            to ``"reranked"``.
        """
        if not results:
            return []

        # Pass the position index as "id" so we can retrieve original metadata
        # after FlashRank reorders — FlashRank knows nothing about our fields.
        passages = [{"id": i, "text": r.chunk_text} for i, r in enumerate(results)]

        reranked = self._ranker.rerank(RerankRequest(query=query, passages=passages))

        return [
            RetrievalResult(
                chunk_text=results[p["id"]].chunk_text,
                score=float(p["score"]),
                metadata=results[p["id"]].metadata,
                retrieval_method="reranked",
            )
            for p in reranked
        ][:top_n]

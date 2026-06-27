from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """A single result returned by any retrieval method.

    All retrieval strategies (dense, sparse, hybrid) return this type so
    downstream code — the reranker, the pipeline, the evaluation layer — never
    needs to know which method produced the result.  The ``retrieval_method``
    field carries that information for dashboard / logging purposes.

    Fields:
        chunk_text:       The raw text of the retrieved chunk.
        score:            Relevance score from the retrieval method.  Cosine
                          similarity for dense (0–1); SPLADE dot product for
                          sparse (unbounded); RRF score for hybrid (0–1).
        metadata:         All chunk-level metadata as a plain dict — includes
                          ``chunk_strategy``, ``chunk_index``, ``chunk_size``,
                          ``parent_source``, ``filename``, ``format``.
        retrieval_method: ``"dense"``, ``"sparse"``, or ``"hybrid"``.  Set by
                          the retriever, not the caller.
    """

    chunk_text: str
    score: float
    metadata: dict
    retrieval_method: str

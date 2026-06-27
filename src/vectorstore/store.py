from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models

from configs.settings import AppConfig
from src.chunking.chunkers import ChunkedDocument
from src.embedding.embedder import SparseEmbedding

logger = logging.getLogger(__name__)

# Named vector keys — referenced in collection creation, upsert, and search.
# Defined once here so a typo doesn't silently create a second vector space.
_DENSE = "dense"
_SPARSE = "sparse"

# Number of candidates fetched from each vector space before RRF fusion.
_PREFETCH_LIMIT = 20


@dataclass
class SearchResult:
    """A single retrieval hit — the only type that leaves this module.

    Nothing outside this file should import from ``qdrant_client`` directly;
    callers work with ``SearchResult`` objects regardless of which search
    method produced them.
    """

    text: str
    score: float
    chunk_strategy: str
    chunk_index: int
    chunk_size: int
    parent_source: str
    filename: str
    format: str


class VectorStore:
    """Qdrant-backed store for hybrid dense + sparse search.

    All ``qdrant_client`` imports are contained here.  Nothing outside this
    file should ever import from ``qdrant_client`` directly — this class is
    the abstraction boundary.

    Usage::

        store = VectorStore()
        store.ensure_collection("rag_fixed")

        dense  = embedder.embed_dense(texts)    # (n, 384) float32
        sparse = embedder.embed_sparse(texts)   # list[SparseVector]
        store.upsert(chunks, dense, sparse, "rag_fixed")

        hits = store.search_hybrid(q_dense, q_sparse, "rag_fixed", limit=5)
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        cfg = config or AppConfig()
        self._config = cfg
        self._client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
        logger.info("VectorStore connected to %s:%s", cfg.qdrant_host, cfg.qdrant_port)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, collection_name: str, recreate: bool = False) -> None:
        """Create the collection if it does not exist.

        Idempotent by default (``recreate=False``): safe to call at the top of
        every indexing run.  Pass ``recreate=True`` for a clean-slate re-index.

        The collection is created with two named vector spaces:
        - ``"dense"``  — cosine similarity, dimension from ``AppConfig``
        - ``"sparse"`` — SPLADE sparse vectors (variable-length, no fixed dim)
        """
        exists = self._client.collection_exists(collection_name)

        if recreate and exists:
            self._client.delete_collection(collection_name)
            logger.info("Deleted existing collection %s for recreate", collection_name)
            exists = False

        if exists:
            logger.info("Collection %s already exists — skipping creation", collection_name)
            return

        self._client.create_collection(
            collection_name,
            vectors_config={
                _DENSE: models.VectorParams(
                    size=self._config.dense_embedding_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                _SPARSE: models.SparseVectorParams()
            },
        )
        logger.info(
            "Created collection %s  (dense_dim=%d, sparse=SPLADE)",
            collection_name,
            self._config.dense_embedding_dim,
        )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(
        self,
        chunks: list[ChunkedDocument],
        dense_vectors: np.ndarray,
        sparse_vectors: list[SparseEmbedding],
        collection_name: str,
        batch_size: int = 100,
    ) -> int:
        """Embed and index *chunks* into *collection_name*.

        Args:
            chunks:           Chunks to index.
            dense_vectors:    Pre-computed dense embeddings, shape ``(len(chunks), dim)``.
            sparse_vectors:   Pre-computed SPLADE vectors, one per chunk.
            collection_name:  Target Qdrant collection (must already exist).
            batch_size:       Points per upsert call. 100 balances throughput
                              and memory; lower if Qdrant reports request-size errors.

        Returns:
            Number of points upserted.
        """
        if len(chunks) != len(dense_vectors) or len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"Length mismatch: chunks={len(chunks)}, "
                f"dense={len(dense_vectors)}, sparse={len(sparse_vectors)}"
            )

        total = 0
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch_chunks = chunks[start:end]
            batch_dense = dense_vectors[start:end]
            batch_sparse = sparse_vectors[start:end]

            points = [
                _make_point(chunk, dense, sparse)
                for chunk, dense, sparse in zip(batch_chunks, batch_dense, batch_sparse)
            ]

            self._client.upsert(collection_name, points=points, wait=True)
            total += len(points)
            logger.info(
                "Upserted %d/%d points into %s",
                total, len(chunks), collection_name,
            )

        return total

    # ------------------------------------------------------------------
    # Corpus access
    # ------------------------------------------------------------------

    def scroll_all(self, collection_name: str, batch: int = 1000) -> list[dict]:
        """Return the payload of every point in *collection_name*.

        Used by BM25Retriever to build its in-memory index at startup.
        Vectors are not fetched — only the payload (chunk text + metadata).

        At millions of points this would be impractical; move to a dedicated
        search engine (Elasticsearch, OpenSearch) at that scale.
        """
        payloads: list[dict] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name,
                limit=batch,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            payloads.extend(r.payload for r in records if r.payload)
            if offset is None:
                break
        logger.info("Scrolled %d points from %s", len(payloads), collection_name)
        return payloads

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_dense(
        self,
        query_vector: list[float],
        collection_name: str,
        limit: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Pure cosine-similarity search over the dense vector space."""
        result = self._client.query_points(
            collection_name,
            query=query_vector,
            using=_DENSE,
            query_filter=_to_filter(filters),
            limit=limit,
            with_payload=True,
        )
        return _to_results(result.points)

    def search_sparse(
        self,
        query_sparse: SparseEmbedding,
        collection_name: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Pure sparse / keyword search over the SPLADE vector space."""
        result = self._client.query_points(
            collection_name,
            query=_to_qdrant_sparse(query_sparse),
            using=_SPARSE,
            limit=limit,
            with_payload=True,
        )
        return _to_results(result.points)

    def search_hybrid(
        self,
        query_dense: list[float],
        query_sparse: SparseEmbedding,
        collection_name: str,
        limit: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: prefetch from both vector spaces, fuse with RRF.

        Qdrant fetches the top-``_PREFETCH_LIMIT`` candidates independently
        from dense and sparse, then applies Reciprocal Rank Fusion internally
        to produce the final ranked list.  This is the production pattern for
        hybrid retrieval.
        """
        result = self._client.query_points(
            collection_name,
            prefetch=[
                models.Prefetch(query=query_dense,                       using=_DENSE,  limit=_PREFETCH_LIMIT),
                models.Prefetch(query=_to_qdrant_sparse(query_sparse),   using=_SPARSE, limit=_PREFETCH_LIMIT),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=_to_filter(filters),
            limit=limit,
            with_payload=True,
        )
        return _to_results(result.points)


# ------------------------------------------------------------------
# Private helpers — not exported
# ------------------------------------------------------------------

def _to_qdrant_sparse(s: SparseEmbedding) -> models.SparseVector:
    """Convert the embedding-layer ``SparseEmbedding`` to a Qdrant ``SparseVector``.

    This is the only place in the codebase that crosses the abstraction
    boundary between the embedding layer and the Qdrant SDK.
    """
    return models.SparseVector(indices=s.indices, values=s.values)


def _make_point(
    chunk: ChunkedDocument,
    dense: np.ndarray,
    sparse: SparseEmbedding,
) -> models.PointStruct:
    """Build a Qdrant PointStruct from a single chunk and its vectors.

    Point IDs are deterministic UUID5s derived from source + strategy + index
    so re-indexing the same document upserts over existing points rather than
    creating duplicates.
    """
    point_id = str(uuid.uuid5(
        uuid.NAMESPACE_OID,
        f"{chunk.parent_source}:{chunk.chunk_strategy}:{chunk.chunk_index}",
    ))
    return models.PointStruct(
        id=point_id,
        vector={
            _DENSE: dense.tolist(),
            _SPARSE: _to_qdrant_sparse(sparse),
        },
        payload={
            "chunk_text":      chunk.text,
            "source_file":     chunk.metadata.filename,
            "chunk_strategy":  chunk.chunk_strategy,
            "chunk_index":     chunk.chunk_index,
            "chunk_size":      chunk.chunk_size,
            "parent_source":   chunk.parent_source,
            "filename":        chunk.metadata.filename,
            "format":          chunk.metadata.format,
            "total_pages":     chunk.metadata.total_pages,
            # page_number is not tracked at chunk level — the chunker
            # operates on concatenated or per-page text without a page cursor.
        },
    )


def _to_filter(filters: dict | None) -> models.Filter | None:
    """Convert a plain ``{"key": value}`` dict to a Qdrant ``Filter``.

    Returns ``None`` if *filters* is empty or None, which Qdrant treats as
    no filter applied.  Calling code never needs to know Qdrant's filter syntax.
    """
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
    )


def _to_results(points: list[models.ScoredPoint]) -> list[SearchResult]:
    """Convert Qdrant ``ScoredPoint`` objects to ``SearchResult`` objects."""
    results = []
    for p in points:
        pl = p.payload or {}
        results.append(SearchResult(
            text=pl.get("chunk_text", ""),
            score=p.score,
            chunk_strategy=pl.get("chunk_strategy", ""),
            chunk_index=pl.get("chunk_index", 0),
            chunk_size=pl.get("chunk_size", 0),
            parent_source=pl.get("parent_source", ""),
            filename=pl.get("filename", ""),
            format=pl.get("format", ""),
        ))
    return results

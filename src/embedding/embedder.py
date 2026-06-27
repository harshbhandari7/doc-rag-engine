from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from configs.settings import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class SparseEmbedding:
    """A sparse vector: parallel lists of active term ids and their weights.

    This is an embedding-layer type — deliberately free of any Qdrant imports.
    ``VectorStore`` converts it to a ``qdrant_client.models.SparseVector`` when
    building points, keeping the abstraction boundary clean.
    """

    indices: list[int]
    values: list[float]


class Embedder:
    """Dense + sparse embeddings for hybrid search.

    Two responsibilities live in one class because both consume the same input
    (a ``list[str]``) and are always called together when indexing a document
    into Qdrant:

    * :meth:`embed_dense`  — SentenceTransformer → ``(n, dim)`` float32 numpy array
    * :meth:`embed_sparse` — fastembed SPLADE → ``list[SparseEmbedding]``

    Usage::

        embedder = Embedder()
        dense  = embedder.embed_dense(["first chunk", "second chunk"])   # (2, 384) float32
        sparse = embedder.embed_sparse(["first chunk", "second chunk"])  # [SparseEmbedding, ...]

    **Loading strategy.** The dense model loads eagerly at construction — it is
    the cheap, always-used path that the semantic chunker depends on. The sparse
    SPLADE model is loaded lazily on the first :meth:`embed_sparse` call (and
    fastembed downloads it on first run), so dense-only callers such as the
    semantic chunker never pay the SPLADE load/download cost.

    Override models via ``APP_DENSE_EMBEDDING_MODEL`` / ``APP_SPARSE_EMBEDDING_MODEL``.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig()
        self._dense_model = SentenceTransformer(self._config.dense_embedding_model)
        self._sparse_model: SparseTextEmbedding | None = None  # lazy — see class docstring

    # ------------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------------

    def embed_dense(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* into dense vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            Float32 array of shape ``(len(texts), dense_embedding_dim)``.
            For ``all-MiniLM-L6-v2`` the dimension is 384.
        """
        return self._dense_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    # ------------------------------------------------------------------
    # Sparse
    # ------------------------------------------------------------------

    def embed_sparse(self, texts: list[str]) -> list[SparseEmbedding]:
        """Encode *texts* into sparse SPLADE vectors.

        Unlike dense vectors, sparse vectors are variable-length: only the
        non-zero vocabulary dimensions are stored as parallel ``indices`` /
        ``values`` lists.  See :class:`SparseEmbedding`.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            One ``SparseEmbedding`` per input string, in the same order.
        """
        model = self._get_sparse_model()
        # Smaller batch than dense: SPLADE runs a full transformer forward pass
        # per text and is markedly slower than MiniLM, so 16 keeps memory and
        # latency in check.
        return [
            SparseEmbedding(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
            for emb in model.embed(texts, batch_size=16)
        ]

    def _get_sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            # The SPLADE model is ~500MB; the first ever call blocks here while
            # fastembed downloads it. Log so an indexing run's pause is explained.
            logger.info(
                "Loading sparse model %s (first run downloads ~500MB)...",
                self._config.sparse_embedding_model,
            )
            self._sparse_model = SparseTextEmbedding(self._config.sparse_embedding_model)
            logger.info("Sparse model ready.")
        return self._sparse_model

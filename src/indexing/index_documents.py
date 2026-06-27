"""Index all documents from data/raw/ into Qdrant using all three chunking strategies.

Run from the project root:

    python -m src.indexing.index_documents

Each run recreates the three collections (rag_fixed, rag_recursive, rag_semantic)
from scratch.  Pass --no-recreate to append to existing collections instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from configs.settings import AppConfig
from src.chunking import FixedSizeChunker, RecursiveChunker, SemanticChunker
from src.embedding import Embedder
from src.ingestion import DocumentLoader, LoadStatus
from src.ingestion.models import LoadedDocument
from src.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Quieten noisy third-party loggers — our own INFO still shows.
for _noisy in ("httpx", "sentence_transformers", "docling"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_documents(data_dir: str) -> list[LoadedDocument]:
    """Load all supported files under *data_dir*, returning only successful loads."""
    loader = DocumentLoader()
    logger.info("Scanning %s for documents...", data_dir)
    all_docs = list(loader.load_directory(data_dir))

    ok   = [d for d in all_docs if d.status != LoadStatus.FAILURE]
    failed = [d for d in all_docs if d.status == LoadStatus.FAILURE]

    if failed:
        for d in failed:
            logger.warning("FAILED  %s  errors=%s", d.metadata.filename, d.errors)

    logger.info(
        "Loaded %d/%d documents  (%d failed)",
        len(ok), len(all_docs), len(failed),
    )
    return ok


def index_strategy(
    strategy_name: str,
    collection_name: str,
    chunker,
    docs: list[LoadedDocument],
    embedder: Embedder,
    store: VectorStore,
    recreate: bool,
) -> dict:
    """Chunk, embed, and upsert one strategy. Returns a summary dict."""
    t0 = time.perf_counter()

    # --- chunk ---
    logger.info("[%s] Chunking %d documents...", strategy_name, len(docs))
    chunks = chunker.chunk(docs)
    logger.info("[%s] Produced %d chunks", strategy_name, len(chunks))

    # --- embed ---
    texts = [c.text for c in chunks]

    logger.info("[%s] Embedding dense (%d texts)...", strategy_name, len(texts))
    dense = embedder.embed_dense(texts)

    logger.info("[%s] Embedding sparse (%d texts)...", strategy_name, len(texts))
    sparse = embedder.embed_sparse(texts)

    # --- index ---
    store.ensure_collection(collection_name, recreate=recreate)

    logger.info("[%s] Upserting into %s...", strategy_name, collection_name)
    n_upserted = store.upsert(chunks, dense, sparse, collection_name)

    elapsed = time.perf_counter() - t0
    return {
        "strategy":   strategy_name,
        "collection": collection_name,
        "chunks":     len(chunks),
        "upserted":   n_upserted,
        "elapsed_s":  round(elapsed, 1),
    }


def print_summary(results: list[dict], total_elapsed: float) -> None:
    header = f"\n{'─'*66}\n  {'STRATEGY':<12} {'COLLECTION':<22} {'CHUNKS':>7} {'POINTS':>7} {'TIME':>7}\n{'─'*66}"
    print(header)
    for r in results:
        print(
            f"  {r['strategy']:<12} {r['collection']:<22} "
            f"{r['chunks']:>7} {r['upserted']:>7} {r['elapsed_s']:>6.1f}s"
        )
    print(f"{'─'*66}")
    print(f"  Total time: {total_elapsed:.1f}s\n")


def main(data_dir: str = "data/raw", recreate: bool = True) -> None:
    cfg = AppConfig()
    t_start = time.perf_counter()

    # --- 1. Load documents (once — shared across all strategies) ---
    docs = load_documents(data_dir)
    if not docs:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)

    # --- 2. Initialise shared components ---
    embedder = Embedder()   # dense model loads now; SPLADE loads lazily on first sparse call
    store    = VectorStore()

    # --- 3. Define strategies in index order ---
    strategies = [
        ("fixed",     cfg.collection_fixed,     FixedSizeChunker(cfg)),
        ("recursive", cfg.collection_recursive, RecursiveChunker(cfg)),
        ("semantic",  cfg.collection_semantic,  SemanticChunker(embedder, cfg)),
    ]

    # --- 4. Index each strategy ---
    results = []
    for strategy_name, collection_name, chunker in strategies:
        logger.info("=" * 60)
        logger.info("Strategy: %s → collection: %s", strategy_name, collection_name)
        logger.info("=" * 60)
        summary = index_strategy(
            strategy_name, collection_name, chunker,
            docs, embedder, store, recreate,
        )
        results.append(summary)
        logger.info(
            "[%s] Done — %d points in %.1fs",
            strategy_name, summary["upserted"], summary["elapsed_s"],
        )

    # --- 5. Summary ---
    print_summary(results, time.perf_counter() - t_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents into Qdrant.")
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Root directory to scan for documents (default: data/raw)",
    )
    parser.add_argument(
        "--no-recreate", action="store_true",
        help="Append to existing collections instead of recreating them",
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir, recreate=not args.no_recreate)

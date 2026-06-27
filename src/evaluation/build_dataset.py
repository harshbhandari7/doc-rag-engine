"""
Build or refresh the evaluation dataset.

Loads data/test_questions.json, runs each question through RAGPipeline,
and writes data/eval_dataset.json with answer, contexts, and metadata.

Run from project root:
    python -m src.evaluation.build_dataset
    python -m src.evaluation.build_dataset --collection rag_semantic
    python -m src.evaluation.build_dataset --no-overwrite   # skip if eval_dataset.json exists
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from configs.settings import AppConfig
from src.embedding.embedder import Embedder
from src.generation import RAGPipeline
from src.vectorstore.store import VectorStore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def build(
    questions_path: str = "data/test_questions.json",
    output_path: str = "data/eval_dataset.json",
    collection: str | None = None,
    overwrite: bool = True,
) -> list[dict]:
    out = Path(output_path)
    if out.exists() and not overwrite:
        print(f"[skip] {output_path} already exists. Pass --overwrite to rebuild.")
        with open(out) as f:
            return json.load(f)

    cfg = AppConfig()
    col = collection or cfg.collection_recursive

    print(f"Loading test questions from {questions_path} ...")
    with open(questions_path) as f:
        questions = json.load(f)
    print(f"  {len(questions)} questions")

    print(f"Building RAGPipeline on collection={col} ...")
    embedder = Embedder()
    store = VectorStore()
    pipeline = RAGPipeline(embedder, store, col, cfg, use_detailed_prompt=False)
    print("  Pipeline ready.\n")

    dataset = []
    ok = err = 0

    for q in questions:
        t0 = time.time()
        result = pipeline.invoke(q["question"])
        elapsed = time.time() - t0

        entry = {
            "id":               q["id"],
            "question":         q["question"],
            "answer":           result["answer"],
            "contexts":         [c.chunk_text for c in result["source_chunks"]],
            "ground_truth":     q["ground_truth"],
            "source_document":  q["source_document"],
            "retrieval_method": result["retrieval_method"],
            "collection":       result["collection"],
            "error":            result["error"],
            "elapsed_s":        round(elapsed, 2),
            "top_retrieved_doc": (
                result["source_chunks"][0].metadata.get("filename", "?")
                if result["source_chunks"] else "none"
            ),
            "top_score": round(result["source_chunks"][0].score, 4) if result["source_chunks"] else 0.0,
        }
        dataset.append(entry)

        if result["error"]:
            err += 1
            status = "ERR"
        else:
            ok += 1
            status = "OK "

        top = entry["top_retrieved_doc"]
        score = entry["top_score"]
        print(f"  {status} {q['id']} | {elapsed:.2f}s | top={top} ({score:.3f}) | {q['question'][:50]}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    correct_source = sum(1 for d in dataset if d["top_retrieved_doc"] == d["source_document"])
    print(f"\nSaved {len(dataset)} entries → {output_path}")
    print(f"  LLM answers OK : {ok}   (errors: {err} — Ollama not running?)")
    print(f"  Top-1 source match: {correct_source}/{len(dataset)}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG evaluation dataset")
    parser.add_argument("--questions", default="data/test_questions.json")
    parser.add_argument("--output",    default="data/eval_dataset.json")
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection name (default: cfg.collection_recursive)")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Skip if output file already exists")
    args = parser.parse_args()
    build(
        questions_path=args.questions,
        output_path=args.output,
        collection=args.collection,
        overwrite=not args.no_overwrite,
    )

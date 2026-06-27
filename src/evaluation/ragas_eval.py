"""
src/evaluation/ragas_eval.py

Two-phase evaluation runner:

  Phase 1 — collect_dataset()
    Runs every question through the RAG pipeline for each (collection, method) pair.
    Saves raw results to disk as JSON *before* handing off to RAGAS so that a
    RAGAS crash or network timeout doesn't require re-running all pipeline calls.

  Phase 2 — run_ragas()
    Converts the saved JSON to ragas.EvaluationDataset, runs the four core metrics,
    and logs per-collection scores to MLflow.

Usage:
    # Full run — collect + evaluate all three collections
    python -m src.evaluation.ragas_eval

    # Skip re-running the pipeline; reuse an existing raw JSON
    python -m src.evaluation.ragas_eval --skip-collect

    # Evaluate a single collection
    python -m src.evaluation.ragas_eval --collections rag_recursive

    # Vary the retrieval method
    python -m src.evaluation.ragas_eval --method dense
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import mlflow
from ragas import evaluate as ragas_evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall

from configs.settings import AppConfig
from src.embedding.embedder import Embedder
from src.evaluation.judge import make_judge
from src.generation import RAGPipeline
from src.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "rag-evaluation"

# Metrics that need only the LLM judge, vs metrics that also need embeddings.
# Kept as constants so the caller can override them without touching this file.
DEFAULT_METRICS_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# Phase 1 — Pipeline collection
# ---------------------------------------------------------------------------

def collect_dataset(
    questions: list[dict],
    pipeline: RAGPipeline,
    collection: str,
    retrieval_method: str,
) -> list[dict]:
    """Run every question through the pipeline and return a list of raw entries.

    Each entry has the four fields RAGAS needs plus diagnostics:
        question, answer, contexts, ground_truth,
        error, elapsed_s, top_retrieved_doc, top_score.

    This function does *not* write to disk — the caller is responsible for
    saving before handing off to RAGAS (see ``run_evaluation``).
    """
    entries: list[dict] = []
    for q in questions:
        t0 = time.time()
        result = pipeline.invoke(q["question"], collection=collection, retrieval_method=retrieval_method)
        elapsed = time.time() - t0

        entry: dict[str, Any] = {
            # RAGAS-required fields
            "question":     q["question"],
            "answer":       result["answer"],
            "contexts":     [c.chunk_text for c in result["source_chunks"]],
            "ground_truth": q["ground_truth"],
            # Diagnostic fields (stripped before RAGAS conversion)
            "id":                q["id"],
            "source_document":   q["source_document"],
            "retrieval_method":  result["retrieval_method"],
            "collection":        result["collection"],
            "error":             result["error"],
            "elapsed_s":         round(elapsed, 2),
            "top_retrieved_doc": (
                result["source_chunks"][0].metadata.get("filename", "?")
                if result["source_chunks"] else "none"
            ),
            "top_score": (
                round(result["source_chunks"][0].score, 4)
                if result["source_chunks"] else 0.0
            ),
        }
        entries.append(entry)

        status = "ERR" if result["error"] else "OK "
        logger.info(
            "%s %s | %.2fs | top=%s (%.3f) | %s",
            status, q["id"], elapsed,
            entry["top_retrieved_doc"], entry["top_score"],
            q["question"][:55],
        )

    return entries


def collect_all(
    questions_path: str,
    cfg: AppConfig,
    collections: list[str],
    retrieval_method: str,
    output_dir: str = "data",
) -> dict[str, list[dict]]:
    """Build one RAGPipeline per collection, collect results for every question,
    and write one JSON file per (collection, method) pair.

    Returns:
        {collection_name: [entry, ...]}
    """
    questions_path_ = Path(questions_path)
    with open(questions_path_) as f:
        questions: list[dict] = json.load(f)

    logger.info("Loaded %d questions from %s", len(questions), questions_path_)

    embedder = Embedder()
    store = VectorStore()

    all_results: dict[str, list[dict]] = {}

    for collection in collections:
        logger.info("Building RAGPipeline for collection=%s ...", collection)
        pipeline = RAGPipeline(embedder, store, collection, cfg, use_detailed_prompt=False)

        logger.info(
            "Collecting %d questions × collection=%s method=%s ...",
            len(questions), collection, retrieval_method,
        )
        entries = collect_dataset(questions, pipeline, collection, retrieval_method)
        all_results[collection] = entries

        # Save raw dataset to disk before touching RAGAS — if RAGAS crashes,
        # you can reload this file and skip re-running the pipeline.
        out_path = Path(output_dir) / f"eval_raw_{collection}_{retrieval_method}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(entries, f, indent=2)

        ok  = sum(1 for e in entries if not e["error"])
        err = sum(1 for e in entries if e["error"])
        src_match = sum(1 for e in entries if e["top_retrieved_doc"] == e["source_document"])
        logger.info(
            "  Saved %s | LLM OK=%d ERR=%d | top-1 source match=%d/%d",
            out_path, ok, err, src_match, len(entries),
        )

    return all_results


# ---------------------------------------------------------------------------
# Phase 2 — RAGAS conversion and evaluation
# ---------------------------------------------------------------------------

def to_ragas_dataset(entries: list[dict]) -> EvaluationDataset:
    """Convert a list of raw pipeline-output dicts to ragas.EvaluationDataset.

    Field mapping (ragas 0.4.x SingleTurnSample schema):
        question      → user_input
        answer        → response
        contexts      → retrieved_contexts
        ground_truth  → reference
    """
    samples = [
        SingleTurnSample(
            user_input=e["question"],
            response=e["answer"],
            retrieved_contexts=e["contexts"],
            reference=e["ground_truth"],
        )
        for e in entries
    ]
    return EvaluationDataset(samples=samples)


def make_metrics() -> list:
    """Return the four core RAGAS metric singleton instances.

    ragas_evaluate() sets .llm and .embeddings on these singletons from its
    own llm= and embeddings= arguments — do not set them here.
    """
    return [faithfulness, answer_relevancy, context_precision, context_recall]


def run_ragas(
    eval_dataset: EvaluationDataset,
    judge_llm: LangchainLLMWrapper,
    judge_emb: LangchainEmbeddingsWrapper,
    raise_exceptions: bool = False,
    max_workers: int = 2,
) -> dict[str, Any]:
    """Run RAGAS evaluation and return a dict of mean metric scores.

    Returns:
        {"faithfulness": float, "answer_relevancy": float, ...}
        Missing metrics (judge API error) are returned as None.
    """
    run_config = RunConfig(max_workers=max_workers, max_retries=10, max_wait=60, timeout=120)
    result = ragas_evaluate(
        dataset=eval_dataset,
        metrics=make_metrics(),
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        raise_exceptions=raise_exceptions,
        show_progress=True,
    )

    # result._repr_dict is {metric_name: mean_score}
    means: dict[str, Any] = dict(result._repr_dict)
    logger.info("RAGAS scores: %s", means)
    return means, result


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_run(
    scores: dict[str, Any],
    params: dict[str, str],
    raw_entries: list[dict],
    run_name: str,
    output_dir: str = "data",
) -> None:
    """Log one (collection, method) evaluation run to MLflow.

    Logs:
        - params: collection, retrieval_method, model, judge_model, n_questions
        - metrics: one float per RAGAS metric
        - artifacts: the raw pipeline output JSON for this run
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        # Log only numeric scores; skip None (metric failed)
        numeric_scores = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
        if numeric_scores:
            mlflow.log_metrics(numeric_scores)

        # Save per-question scores as a JSON artifact
        artifact_path = (
            Path(output_dir)
            / f"mlflow_scores_{params['collection']}_{params['retrieval_method']}.json"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump(
                {
                    "params": params,
                    "mean_scores": scores,
                    "per_question": raw_entries,
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(artifact_path))

    logger.info("MLflow run '%s' logged: %s", run_name, numeric_scores)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_evaluation(
    questions_path: str = "data/test_questions.json",
    output_dir: str = "data",
    collections: list[str] | None = None,
    retrieval_method: str = "hybrid_rrf",
    skip_collect: bool = False,
    raise_exceptions: bool = False,
    cfg: AppConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the full evaluation pipeline across all specified collections.

    Args:
        questions_path:   Path to test_questions.json.
        output_dir:       Directory for raw JSON outputs and MLflow artifacts.
        collections:      List of Qdrant collections to evaluate. Defaults to all three.
        retrieval_method: Retrieval strategy passed to pipeline.invoke().
        skip_collect:     If True, load existing eval_raw_*.json files instead of
                          re-running the pipeline. Useful when re-running only RAGAS.
        raise_exceptions: Whether RAGAS should re-raise evaluation errors.
        cfg:              AppConfig instance; constructed from env if None.

    Returns:
        {collection_name: {"scores": {...}, "n_questions": int}}
    """
    cfg = cfg or AppConfig()
    if collections is None:
        collections = [cfg.collection_fixed, cfg.collection_recursive, cfg.collection_semantic]

    # Phase 1 — collect (or load from disk)
    if skip_collect:
        logger.info("--skip-collect: loading existing raw JSON files from %s", output_dir)
        all_results: dict[str, list[dict]] = {}
        for col in collections:
            path = Path(output_dir) / f"eval_raw_{col}_{retrieval_method}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Raw dataset not found at {path}. "
                    "Run without --skip-collect first."
                )
            with open(path) as f:
                all_results[col] = json.load(f)
            logger.info("  Loaded %d entries from %s", len(all_results[col]), path)
    else:
        all_results = collect_all(
            questions_path=questions_path,
            cfg=cfg,
            collections=collections,
            retrieval_method=retrieval_method,
            output_dir=output_dir,
        )

    # Phase 2 — RAGAS + MLflow
    logger.info("Building RAGAS judge (model=%s) ...", cfg.ragas_judge_model)
    judge_llm, judge_emb = make_judge(cfg)

    summary: dict[str, dict[str, Any]] = {}

    for collection, entries in all_results.items():
        logger.info(
            "Running RAGAS on collection=%s method=%s (%d entries) ...",
            collection, retrieval_method, len(entries),
        )
        eval_dataset = to_ragas_dataset(entries)
        scores, _ = run_ragas(
            eval_dataset, judge_llm, judge_emb,
            raise_exceptions=raise_exceptions,
            max_workers=cfg.ragas_max_workers,
        )

        params = {
            "collection":        collection,
            "retrieval_method":  retrieval_method,
            "ollama_model":      cfg.ollama_model,
            "judge_model":       cfg.ragas_judge_model,
            "embed_model":       cfg.dense_embedding_model,
            "n_questions":       str(len(entries)),
            "rerank_top_n":      str(cfg.rerank_top_n),
        }
        run_name = f"{collection}__{retrieval_method}"
        log_run(scores, params, entries, run_name, output_dir)

        summary[collection] = {
            "scores":      scores,
            "n_questions": len(entries),
            "params":      params,
        }

    # Write consolidated summary
    summary_path = Path(output_dir) / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", summary_path)

    # Print table to stdout
    _print_summary(summary, retrieval_method)
    return summary


def _print_summary(summary: dict[str, dict], retrieval_method: str) -> None:
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    header = f"{'Collection':<22}" + "".join(f"{m[:14]:>16}" for m in metric_names)
    sep = "-" * len(header)
    print(f"\n=== RAGAS Evaluation Summary (method={retrieval_method}) ===")
    print(header)
    print(sep)
    for col, data in summary.items():
        scores = data["scores"]
        row = f"{col:<22}" + "".join(
            f"{scores.get(m, float('nan')):>16.4f}" for m in metric_names
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run RAG evaluation with RAGAS + MLflow")
    parser.add_argument("--questions", default="data/test_questions.json")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--collections", nargs="+",
        default=None,
        help="Qdrant collections to evaluate (default: all three)",
    )
    parser.add_argument(
        "--method", default="hybrid_rrf",
        choices=["hybrid_rrf", "hybrid_weighted", "dense", "bm25"],
        help="Retrieval method (default: hybrid_rrf)",
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Skip pipeline runs; load existing eval_raw_*.json files",
    )
    parser.add_argument(
        "--raise-exceptions", action="store_true",
        help="Propagate RAGAS metric errors instead of returning NaN",
    )
    args = parser.parse_args()

    run_evaluation(
        questions_path=args.questions,
        output_dir=args.output_dir,
        collections=args.collections,
        retrieval_method=args.method,
        skip_collect=args.skip_collect,
        raise_exceptions=args.raise_exceptions,
    )

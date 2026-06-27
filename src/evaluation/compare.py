"""
src/evaluation/compare.py

Full 3-collection × 4-retrieval-method comparison matrix.

12 MLflow runs total — one per (collection, variant). Each run logs:
  params   — collection, method, rerank, chunk_size, model names, top_k
  metrics  — 4 RAGAS scores + mean retrieval latency + mean generation latency
  artifacts — raw dataset JSON + per-question scores CSV

Variants (4):
  dense             — MiniLM bi-encoder only, no cross-encoder rerank
  bm25              — Okapi BM25 term overlap only, no rerank
  hybrid_rrf        — RRF fusion of dense + BM25, no rerank
  hybrid_rrf+rerank — RRF fusion then FlashRank cross-encoder (production config)

Collections (3):
  rag_fixed      — hard 512-char character splits
  rag_recursive  — paragraph-aware recursive splits
  rag_semantic   — sentence-transformer breakpoint splits

Run from project root:
    python -m src.evaluation.compare
    python -m src.evaluation.compare --skip-collect   # reuse saved JSON, re-run RAGAS
    python -m src.evaluation.compare --variants hybrid_rrf hybrid_rrf+rerank
    python -m src.evaluation.compare --dry-run        # print matrix, exit
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow

from configs.settings import AppConfig
from src.embedding.embedder import Embedder
from src.evaluation.judge import make_judge
from src.evaluation.ragas_eval import make_metrics, to_ragas_dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.generation.pipeline import RAGPipeline
from src.generation.prompts import format_context
from src.vectorstore.store import VectorStore
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    """One retrieval configuration to evaluate."""
    label: str          # used in MLflow run name and output file names
    method: str         # passed to pipeline._retrieve()
    rerank: bool        # whether to apply the cross-encoder after retrieval


ALL_VARIANTS: list[Variant] = [
    Variant(label="dense",              method="dense",       rerank=False),
    Variant(label="bm25",               method="bm25",        rerank=False),
    Variant(label="hybrid_rrf",         method="hybrid_rrf",  rerank=False),
    Variant(label="hybrid_rrf+rerank",  method="hybrid_rrf",  rerank=True),
]

VARIANT_BY_LABEL: dict[str, Variant] = {v.label: v for v in ALL_VARIANTS}


# ---------------------------------------------------------------------------
# Per-question data collection
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    """All data captured for one question in one (collection, variant) run."""
    id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    source_document: str
    retrieval_latency_s: float
    generation_latency_s: float
    top_retrieved_doc: str
    top_score: float
    error: str | None


def _run_question(
    q: dict,
    pipeline: RAGPipeline,
    collection: str,
    variant: Variant,
) -> QuestionResult:
    """Run one question through retrieval + generation with separate timing."""
    # --- retrieval (timed separately) ---
    t0 = time.perf_counter()
    chunks = pipeline._retrieve(
        q["question"], collection, variant.method, rerank=variant.rerank
    )
    retrieval_latency = time.perf_counter() - t0

    if not chunks:
        return QuestionResult(
            id=q["id"], question=q["question"],
            answer="No relevant context was found in the documents for this query.",
            contexts=[], ground_truth=q["ground_truth"],
            source_document=q["source_document"],
            retrieval_latency_s=retrieval_latency, generation_latency_s=0.0,
            top_retrieved_doc="none", top_score=0.0, error=None,
        )

    chunks = pipeline._truncate_to_limit(chunks, q["question"])
    context = format_context(chunks)

    # --- generation (timed separately) ---
    t0 = time.perf_counter()
    try:
        answer = pipeline._llm_chain.invoke({"context": context, "question": q["question"]})
        error: str | None = None
    except Exception as exc:
        answer = "Generation failed: the language model did not return a response."
        error = str(exc)
    generation_latency = time.perf_counter() - t0

    return QuestionResult(
        id=q["id"], question=q["question"],
        answer=answer,
        contexts=[c.chunk_text for c in chunks],
        ground_truth=q["ground_truth"],
        source_document=q["source_document"],
        retrieval_latency_s=round(retrieval_latency, 4),
        generation_latency_s=round(generation_latency, 4),
        top_retrieved_doc=chunks[0].metadata.get("filename", "?"),
        top_score=round(chunks[0].score, 4),
        error=error,
    )


def collect_run(
    questions: list[dict],
    pipeline: RAGPipeline,
    collection: str,
    variant: Variant,
    output_dir: Path,
) -> list[QuestionResult]:
    """Collect all questions for one (collection, variant) pair.

    Saves raw JSON immediately so RAGAS can be re-run without re-calling the
    pipeline (use --skip-collect).
    """
    results: list[QuestionResult] = []
    for q in questions:
        r = _run_question(q, pipeline, collection, variant)
        results.append(r)
        status = "ERR" if r.error else "OK "
        logger.info(
            "%s %s | ret=%.3fs gen=%.3fs | top=%s(%.3f) | %s",
            status, r.id,
            r.retrieval_latency_s, r.generation_latency_s,
            r.top_retrieved_doc, r.top_score,
            r.question[:50],
        )

    # Persist before RAGAS
    raw_path = output_dir / f"raw_{collection}_{variant.label}.json"
    with open(raw_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    logger.info("Saved %d entries → %s", len(results), raw_path)
    return results


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def _to_csv(results: list[QuestionResult], ragas_scores: list[dict]) -> str:
    """Build a per-question CSV string combining retrieval data and RAGAS scores."""
    metric_names = list(ragas_scores[0].keys()) if ragas_scores else []
    header = [
        "id", "question", "source_document", "top_retrieved_doc", "top_score",
        "retrieval_latency_s", "generation_latency_s", "error",
    ] + metric_names

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore")
    writer.writeheader()
    for r, scores in zip(results, ragas_scores or [{}] * len(results)):
        row = {
            "id": r.id, "question": r.question[:80],
            "source_document": r.source_document,
            "top_retrieved_doc": r.top_retrieved_doc,
            "top_score": r.top_score,
            "retrieval_latency_s": r.retrieval_latency_s,
            "generation_latency_s": r.generation_latency_s,
            "error": r.error or "",
        }
        row.update(scores)
        writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# One MLflow run
# ---------------------------------------------------------------------------

def _log_one_run(
    collection: str,
    variant: Variant,
    results: list[QuestionResult],
    ragas_means: dict[str, float],
    ragas_per_question: list[dict],
    cfg: AppConfig,
    output_dir: Path,
) -> None:
    run_name = f"{collection}__{variant.label}"

    ok  = sum(1 for r in results if not r.error)
    err = sum(1 for r in results if r.error)
    mean_ret = sum(r.retrieval_latency_s for r in results) / len(results)
    mean_gen = sum(r.generation_latency_s for r in results) / len(results)
    src_match = sum(1 for r in results if r.top_retrieved_doc == r.source_document)

    params = {
        "collection":        collection,
        "retrieval_method":  variant.method,
        "rerank":            str(variant.rerank),
        "variant_label":     variant.label,
        "chunk_size":        str(cfg.chunk_size),
        "chunk_overlap":     str(cfg.chunk_overlap),
        "embedding_model":   cfg.dense_embedding_model,
        "llm_model":         cfg.ollama_model,
        "judge_model":       cfg.ragas_judge_model,
        "retrieval_limit":   str(cfg.retrieval_limit),
        "rerank_top_n":      str(cfg.rerank_top_n),
        "n_questions":       str(len(results)),
        "n_llm_errors":      str(err),
    }

    metrics: dict[str, float] = {
        "mean_retrieval_latency_s":  mean_ret,
        "mean_generation_latency_s": mean_gen,
        "top1_source_match_rate":    src_match / len(results),
        "llm_success_rate":          ok / len(results),
    }
    # RAGAS scores — skip None/NaN
    for k, v in ragas_means.items():
        if isinstance(v, (int, float)) and v == v:  # NaN check
            metrics[k] = float(v)

    with mlflow.start_run(run_name=run_name):
        # Log params first — if RAGAS crashes later the run still records config
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Artifact 1: raw collected dataset JSON
        raw_path = output_dir / f"raw_{collection}_{variant.label}.json"
        if raw_path.exists():
            mlflow.log_artifact(str(raw_path), artifact_path="raw_data")

        # Artifact 2: per-question scores CSV
        csv_str = _to_csv(results, ragas_per_question)
        csv_path = output_dir / f"scores_{collection}_{variant.label}.csv"
        csv_path.write_text(csv_str)
        mlflow.log_artifact(str(csv_path), artifact_path="scores")

    logger.info(
        "MLflow run '%s' logged | faithfulness=%.3f answer_relevancy=%.3f "
        "ret=%.3fs gen=%.3fs",
        run_name,
        ragas_means.get("faithfulness", float("nan")),
        ragas_means.get("answer_relevancy", float("nan")),
        mean_ret, mean_gen,
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_comparison(
    questions_path: str = "data/test_questions.json",
    output_dir: str = "data/compare",
    collections: list[str] | None = None,
    variants: list[str] | None = None,
    skip_collect: bool = False,
    raise_exceptions: bool = False,
    cfg: AppConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the full comparison matrix and return aggregated results.

    Args:
        questions_path:   Path to test_questions.json.
        output_dir:       Directory for all output files and MLflow artifacts.
        collections:      Collections to evaluate (default: all three).
        variants:         Variant labels to run (default: all four).
        skip_collect:     If True, load existing raw_*.json files instead of
                          re-running the pipeline.
        raise_exceptions: Whether RAGAS should re-raise metric errors.
        cfg:              AppConfig; constructed from env if None.

    Returns:
        Nested dict: {collection: {variant_label: {"metrics": ..., "n": ...}}}
    """
    cfg = cfg or AppConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if collections is None:
        collections = [cfg.collection_fixed, cfg.collection_recursive, cfg.collection_semantic]
    selected_variants = (
        [VARIANT_BY_LABEL[v] for v in variants]
        if variants else ALL_VARIANTS
    )

    # MLflow setup — SQLite backend, no server required
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    logger.info(
        "MLflow tracking → %s  experiment → %s",
        cfg.mlflow_tracking_uri, cfg.mlflow_experiment_name,
    )

    with open(questions_path) as f:
        questions: list[dict] = json.load(f)
    logger.info("Loaded %d questions from %s", len(questions), questions_path)

    # Build RAGAS judge once — same judge for all 12 runs
    logger.info("Building RAGAS judge (model=%s) ...", cfg.ragas_judge_model)
    judge_llm, judge_emb = make_judge(cfg)

    # Build one RAGPipeline per collection (BM25 index builds once per collection)
    pipelines: dict[str, RAGPipeline] = {}
    embedder = Embedder()
    store = VectorStore()
    for col in collections:
        logger.info("Building RAGPipeline for %s ...", col)
        pipelines[col] = RAGPipeline(embedder, store, col, cfg, use_detailed_prompt=False)

    summary: dict[str, dict[str, Any]] = {}

    for col in collections:
        summary[col] = {}
        pipeline = pipelines[col]

        for variant in selected_variants:
            raw_path = out / f"raw_{col}_{variant.label}.json"

            # Phase 1 — collect
            if skip_collect and raw_path.exists():
                logger.info("skip-collect: loading %s", raw_path)
                with open(raw_path) as f:
                    raw = json.load(f)
                results = [QuestionResult(**r) for r in raw]
            else:
                logger.info(
                    "Collecting %d questions | col=%s variant=%s ...",
                    len(questions), col, variant.label,
                )
                results = collect_run(questions, pipeline, col, variant, out)

            # Phase 2 — RAGAS
            logger.info(
                "Running RAGAS | col=%s variant=%s ...", col, variant.label,
            )
            eval_entries = [
                {
                    "question":     r.question,
                    "answer":       r.answer,
                    "contexts":     r.contexts,
                    "ground_truth": r.ground_truth,
                }
                for r in results
            ]
            eval_dataset = to_ragas_dataset(eval_entries)
            run_config = RunConfig(
                max_workers=cfg.ragas_max_workers,
                max_retries=10,
                max_wait=60,
                timeout=120,
            )
            ragas_result = ragas_evaluate(
                dataset=eval_dataset,
                metrics=make_metrics(),
                llm=judge_llm,
                embeddings=judge_emb,
                run_config=run_config,
                raise_exceptions=raise_exceptions,
                show_progress=True,
            )
            ragas_means: dict[str, float] = dict(ragas_result._repr_dict)
            ragas_per_q: list[dict] = ragas_result.scores

            # Phase 3 — log to MLflow
            _log_one_run(col, variant, results, ragas_means, ragas_per_q, cfg, out)

            summary[col][variant.label] = {
                "metrics": ragas_means,
                "mean_retrieval_latency_s": sum(r.retrieval_latency_s for r in results) / len(results),
                "mean_generation_latency_s": sum(r.generation_latency_s for r in results) / len(results),
                "n_questions": len(results),
                "n_errors": sum(1 for r in results if r.error),
            }

    # Write consolidated summary JSON
    summary_path = out / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary → %s", summary_path)

    _print_table(summary, selected_variants)
    return summary


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_table(
    summary: dict[str, dict[str, Any]],
    variants: list[Variant],
) -> None:
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    latency_cols = ["mean_retrieval_latency_s", "mean_generation_latency_s"]

    col_w = 24
    met_w = 18

    print("\n" + "=" * 110)
    print("RAGAS Comparison Matrix")
    print("=" * 110)
    header = f"{'Collection / Variant':<{col_w}}" + "".join(
        f"{m[:met_w-2]:>{met_w}}" for m in metric_cols + ["ret_lat_s", "gen_lat_s"]
    )
    print(header)
    print("-" * 110)

    for col, var_results in summary.items():
        for v in variants:
            if v.label not in var_results:
                continue
            data = var_results[v.label]
            label = f"{col[:14]}/{v.label[:8]}"
            scores = data.get("metrics", {})
            row = f"{label:<{col_w}}"
            for m in metric_cols:
                val = scores.get(m, float("nan"))
                row += f"{val:>{met_w}.4f}" if isinstance(val, float) else f"{'N/A':>{met_w}}"
            row += f"{data['mean_retrieval_latency_s']:>{met_w}.3f}"
            row += f"{data['mean_generation_latency_s']:>{met_w}.3f}"
            print(row)
        print("-" * 110)

    print(f"\nMLflow UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")
    print("Then open http://localhost:5000 and select 'rag_chunking_comparison'\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="RAG comparison matrix — 3 collections × 4 variants")
    parser.add_argument("--questions",   default="data/test_questions.json")
    parser.add_argument("--output-dir",  default="data/compare")
    parser.add_argument(
        "--collections", nargs="+", default=None,
        help="Qdrant collections (default: all three)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        choices=list(VARIANT_BY_LABEL),
        help=f"Variants to run (default: all). Choices: {list(VARIANT_BY_LABEL)}",
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Skip pipeline runs — load existing raw_*.json files",
    )
    parser.add_argument(
        "--raise-exceptions", action="store_true",
        help="Re-raise RAGAS metric errors instead of returning NaN",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the comparison matrix without running anything",
    )
    args = parser.parse_args()

    if args.dry_run:
        cfg = AppConfig()
        cols = args.collections or [cfg.collection_fixed, cfg.collection_recursive, cfg.collection_semantic]
        vs   = [VARIANT_BY_LABEL[v] for v in args.variants] if args.variants else ALL_VARIANTS
        print(f"\nDry-run: {len(cols)} collections × {len(vs)} variants = {len(cols) * len(vs)} MLflow runs")
        for col in cols:
            for v in vs:
                print(f"  {col:<26}  {v.label}")
        raise SystemExit(0)

    run_comparison(
        questions_path=args.questions,
        output_dir=args.output_dir,
        collections=args.collections,
        variants=args.variants,
        skip_collect=args.skip_collect,
        raise_exceptions=args.raise_exceptions,
    )

"""
Streamlit RAG Dashboard — doc-rag-engine

Three tabs:
  Ask              — stream a live answer from one collection + method
  Compare          — run the same question across all three collections side-by-side
  Evaluation       — interactive charts from the 3×4 RAGAS evaluation matrix

Run:
    source venv/bin/activate
    streamlit run app.py

# ---------------------------------------------------------------------------
# Caching strategy
#
# Streamlit reruns the entire script on every user interaction (button click,
# dropdown change, text input). Without caching, we'd reload the embedding
# model and rebuild the BM25 index on every keystroke.
#
# @st.cache_resource  — for objects that CANNOT be serialised (ML models,
#                       network connections). Lives for the lifetime of the
#                       server process, shared across all browser sessions.
#                       Used for: Embedder, VectorStore, RAGPipeline.
#                       These are slow to construct (~5–30 s) and stateless
#                       for reading, so sharing across sessions is safe.
#
# @st.cache_data      — for serialisable return values (dicts, DataFrames).
#                       Keyed by function arguments; safe to pickle and store.
#                       Used for: evaluation summary (disk read that doesn't
#                       change between compare.py runs).
#                       NOT used for pipeline.invoke() — every question is
#                       different; caching would return stale answers.
#
# Never cached:
#   - pipeline.stream() / pipeline.invoke() calls in the Ask and Compare tabs
#     (each call is a live query with a unique question)
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.generation.prompts import format_context

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="doc-rag-engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_LABELS = {
    "rag_recursive": "rag_recursive ✨ (recommended)",
    "rag_fixed":     "rag_fixed",
    "rag_semantic":  "rag_semantic",
}

COLLECTION_NAMES = list(COLLECTION_LABELS.keys())

METHOD_LABELS = {
    "hybrid_rrf":      "hybrid_rrf + rerank (recommended)",
    "dense":           "dense + rerank",
    "bm25":            "bm25 + rerank",
    "hybrid_weighted": "hybrid_weighted + rerank",
}

METRIC_LABELS = {
    "faithfulness":     "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":   "Context Recall",
}

METRIC_COLORS = {
    "faithfulness":     "#636EFA",
    "answer_relevancy": "#EF553B",
    "context_precision": "#00CC96",
    "context_recall":   "#AB63FA",
}

# Short display names for the three chunking strategies (used in compare tab columns)
COL_LABELS = {
    "rag_fixed":     "Fixed-size",
    "rag_recursive": "Recursive",
    "rag_semantic":  "Semantic",
}

# Display order for the compare tab: simplest → most complex chunking
_COMPARE_ORDER = ["rag_fixed", "rag_recursive", "rag_semantic"]

# Maps sidebar method key → variant key present in comparison_summary.json.
# The dashboard always applies the cross-encoder reranker, so hybrid_rrf maps
# to "hybrid_rrf+rerank" (the evaluated reranked variant) as the closest match.
_EVAL_METHOD_MAP: dict = {
    "hybrid_rrf":      "hybrid_rrf+rerank",
    "hybrid_weighted": None,   # not in the evaluation matrix
    "dense":           "dense",
    "bm25":            "bm25",
}

SUMMARY_PATH = Path("data/compare/comparison_summary.json")

# ---------------------------------------------------------------------------
# Session state — initialise once; widgets with key= sync automatically.
#
# Storing selections here means they persist as the user switches tabs.
# Without this, each tab would reset to its own default on every Streamlit
# rerun (every click or keystroke triggers a full script re-execution).
# ---------------------------------------------------------------------------
_SS_DEFAULTS: dict = {
    "collection": "rag_recursive",  # evaluation-backed best collection
    "method":     "hybrid_rrf",     # evaluation-backed best first-stage method
    "top_k":      5,                # matches cfg.rerank_top_n default
    "query_text": "",               # pre-fillable via example question buttons
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Cached resources — constructed once per server process, not per rerun
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedder and vector store…")
def _get_shared() -> tuple:
    from configs.settings import AppConfig
    from src.embedding.embedder import Embedder
    from src.vectorstore.store import VectorStore
    cfg = AppConfig()
    return Embedder(), VectorStore(), cfg


@st.cache_resource(show_spinner="Building pipeline (BM25 index + cross-encoder)…")
def _get_pipeline(collection: str):
    from src.generation import RAGPipeline
    embedder, store, cfg = _get_shared()
    return RAGPipeline(embedder, store, collection, cfg, use_detailed_prompt=True)


# ---------------------------------------------------------------------------
# Health checks — cached so they don't re-fire on every widget interaction,
# but with a short TTL so recovery (starting Docker, fixing the key) is
# picked up within one cache window without a server restart.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _qdrant_reachable() -> bool:
    """Ping Qdrant with a 3-second timeout. Re-checked every 30 seconds."""
    from qdrant_client import QdrantClient
    from configs.settings import AppConfig
    cfg = AppConfig()
    try:
        QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port, timeout=3).get_collections()
        return True
    except Exception:
        return False


@st.cache_data(ttl=60)
def _collection_point_count(collection: str) -> int:
    """Return the number of points in a Qdrant collection (0 on any error)."""
    try:
        _, store, _ = _get_shared()
        info = store._client.get_collection(collection)
        return info.points_count or 0
    except Exception:
        return 0


def _generation_error_msg(exc: Exception) -> str:
    """Return a user-friendly error string with actionable guidance."""
    raw = str(exc).lower()
    if "401" in raw or "unauthorized" in raw:
        return (
            "**API authentication failed (401 Unauthorized).**  \n"
            "Check `OLLAMA_API_KEY` and `OLLAMA_BASE_URL` in your `.env` file."
        )
    if "429" in raw or "rate limit" in raw or "too many" in raw:
        return (
            "**Rate limit hit (429 Too Many Requests).**  \n"
            "The free-tier Ollama API has per-minute limits. Wait a moment and try again."
        )
    if "connect" in raw or "refused" in raw or "unreachable" in raw:
        return (
            "**Cannot reach the LLM API.**  \n"
            "Check that `OLLAMA_BASE_URL` in `.env` points to a running Ollama instance."
        )
    if "timeout" in raw or "timed out" in raw:
        return (
            "**Request timed out.**  \n"
            "The LLM took too long to respond. "
            "Try a shorter question or increase `REQUEST_TIMEOUT` in `.env`."
        )
    return f"**Generation failed.**  \n`{exc}`"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_expander(chunks: list, label: str = "Source chunks") -> None:
    if not chunks:
        return
    with st.expander(f"📄 {label} ({len(chunks)} chunks)", expanded=False):
        for i, c in enumerate(chunks, 1):
            fname  = c.metadata.get("filename", "unknown")
            cidx   = c.metadata.get("chunk_index", "?")
            strat  = c.metadata.get("chunk_strategy", "")
            score  = f"{c.score:.4f}" if c.score is not None else "—"
            st.markdown(
                f"**[{i}]** `{fname}` · chunk {cidx}"
                + (f" · *{strat}*" if strat else "")
                + f" · score {score}"
            )
            st.caption(c.chunk_text[:400] + ("…" if len(c.chunk_text) > 400 else ""))
            if i < len(chunks):
                st.divider()


@st.cache_data
def _load_eval_data() -> tuple[dict, pd.DataFrame] | None:
    """Read comparison_summary.json and build the evaluation DataFrame.

    Cached with @st.cache_data: the file only changes when compare.py reruns.
    Streamlit's 'Clear cache' menu entry (hamburger → Settings) picks up new
    results without restarting the server.
    """
    if not SUMMARY_PATH.exists():
        return None
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    rows = []
    for col, variants in summary.items():
        for variant, data in variants.items():
            row = {
                "collection": col,
                "variant":    variant,
            }
            row.update(data["metrics"])
            row["retrieval_latency_s"]  = data.get("mean_retrieval_latency_s", None)
            row["generation_latency_s"] = data.get("mean_generation_latency_s", None)
            rows.append(row)

    return summary, pd.DataFrame(rows)


@st.cache_data
def _load_mlflow_runs() -> pd.DataFrame | None:
    """Query MLflow for the latest run per (collection, variant).

    Uses experiment_ids (MLflow 3.x requires IDs, not names).
    Deduplicates by keeping the most recent run per configuration —
    re-runs of the same config produce duplicate rows in MLflow.

    Returns a tidy DataFrame or None if the experiment doesn't exist yet.
    """
    import mlflow
    from configs.settings import AppConfig
    cfg = AppConfig()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    try:
        exp = mlflow.get_experiment_by_name(cfg.mlflow_experiment_name)
        if exp is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
        )
    except Exception:
        return None

    if runs.empty:
        return None

    rename = {
        "params.collection":                 "collection",
        "params.variant_label":              "variant",
        "params.rerank":                     "reranked",
        "metrics.faithfulness":              "faithfulness",
        "metrics.answer_relevancy":          "answer_relevancy",
        "metrics.context_precision":         "context_precision",
        "metrics.context_recall":            "context_recall",
        "metrics.mean_retrieval_latency_s":  "retrieval_latency_s",
        "metrics.mean_generation_latency_s": "generation_latency_s",
    }
    keep = {k: v for k, v in rename.items() if k in runs.columns}
    tidy = runs[list(keep) + ["run_id", "start_time"]].rename(columns=keep)

    # Keep only the most recent run per (collection, variant)
    tidy = (
        tidy
        .sort_values("start_time", ascending=False)
        .drop_duplicates(subset=["collection", "variant"])
        .reset_index(drop=True)
    )
    tidy["total_latency_s"] = (
        tidy["retrieval_latency_s"].fillna(0) + tidy["generation_latency_s"].fillna(0)
    )
    return tidy


@st.cache_data
def _load_per_question_df() -> pd.DataFrame | None:
    """Load and concatenate all per-question CSV files from data/compare/.

    Each file is scores_{collection}_{variant}.csv.  Collection and variant are
    parsed from the filename and inserted as columns — the CSVs themselves only
    carry question-level scores.
    """
    compare_dir = Path("data/compare")
    if not compare_dir.exists():
        return None

    dfs = []
    for f in sorted(compare_dir.glob("scores_*.csv")):
        stem = f.stem[len("scores_"):]  # e.g. "rag_recursive_hybrid_rrf+rerank"
        collection = None
        for col in ["rag_fixed", "rag_recursive", "rag_semantic"]:
            if stem.startswith(col + "_"):
                collection = col
                variant    = stem[len(col) + 1:]
                break
        if collection is None:
            continue
        df = pd.read_csv(f)
        df.insert(0, "collection", collection)
        df.insert(1, "variant",    variant)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None


# ---------------------------------------------------------------------------
# Retrieval helper — respects sidebar top-K slider
# ---------------------------------------------------------------------------

def _retrieve_with_top_k(
    pipeline,
    question: str,
    collection: str,
    method: str,
    top_k: int,
) -> list:
    """First-stage retrieval (20 candidates) → cross-encoder rerank → top_k.

    Bypasses pipeline._retrieve() so the sidebar slider can override the
    chunk count without mutating the shared AppConfig. The cross-encoder
    always scores all 20 first-stage candidates; top_k only controls how many
    of those scored chunks reach the LLM prompt.
    """
    limit = pipeline._cfg.retrieval_limit  # 20 — fixed first-stage budget
    if method == "hybrid_rrf":
        candidates = pipeline._hybrid.retrieve_rrf(question, collection, limit=limit)
    elif method == "hybrid_weighted":
        candidates = pipeline._hybrid.retrieve_weighted(question, collection, limit=limit)
    elif method == "dense":
        candidates = pipeline._dense.retrieve(question, collection, limit=limit)
    else:  # bm25
        candidates = pipeline._bm25.retrieve(question, limit=limit)
    chunks = pipeline._reranker.rerank(question, candidates, top_n=top_k)
    return pipeline._truncate_to_limit(chunks, question)


# ---------------------------------------------------------------------------
# Tab: Ask
# ---------------------------------------------------------------------------

# Real questions from data/test_questions.json — chosen to span document types
_EXAMPLE_QUESTIONS = [
    "How does RAGAS compute the faithfulness metric for a generated answer?",
    "What was the real GDP growth rate in the fourth quarter of 2025?",
    "How does natural selection act on variations among individuals according to Darwin?",
]


def tab_ask() -> None:
    st.header("Ask a question")
    st.caption("Collection, retrieval method, and top-K are set in the sidebar.")

    # on_click callbacks run before the script re-executes, so the text_area
    # below picks up the new value on the same rerun.
    def _prefill(text: str) -> None:
        st.session_state["query_text"] = text

    # ---- question input -------------------------------------------------------
    question = st.text_area(
        "Question",
        placeholder="Ask anything about the indexed documents…",
        height=80,
        key="query_text",
    )

    btn_col, _ = st.columns([1, 5])
    run = btn_col.button("Run query", type="primary", disabled=not question.strip())

    # ---- empty state: show examples before first query -----------------------
    if not question.strip():
        st.caption("Not sure what to ask? Try one of these:")
        ex_cols = st.columns(len(_EXAMPLE_QUESTIONS))
        for col, ex in zip(ex_cols, _EXAMPLE_QUESTIONS):
            col.button(ex, on_click=_prefill, args=(ex,), use_container_width=True)
        return  # nothing more to render until the user submits

    if not run:
        return  # question typed but button not clicked yet

    # ---- phase 1: retrieval (blocking) ----------------------------------------
    collection = st.session_state["collection"]
    method     = st.session_state["method"]
    top_k      = st.session_state["top_k"]
    q          = question.strip()

    # Guard: empty collection means indexing was never run
    if _collection_point_count(collection) == 0:
        st.warning(
            f"Collection `{collection}` has no indexed documents.  \n"
            "Run `python -m src.indexing.index_documents` to populate it, "
            "then refresh."
        )
        return

    pipeline = _get_pipeline(collection)

    t_ret_start = time.perf_counter()
    with st.spinner("Retrieving and reranking…"):
        chunks = _retrieve_with_top_k(pipeline, q, collection, method, top_k)
    retrieval_s = time.perf_counter() - t_ret_start

    if not chunks:
        st.warning("No relevant context found. Try rephrasing the question.")
        return

    context = format_context(chunks)

    # ---- phase 2: two-column layout -------------------------------------------
    # Chunks are populated first (instant); answer streams into the left column
    # while the user can already read which sources were retrieved.
    answer_col, chunks_col = st.columns([3, 2])

    with chunks_col:
        st.subheader(f"Sources ({len(chunks)})")
        for i, c in enumerate(chunks, 1):
            fname    = c.metadata.get("filename", "unknown")
            cidx     = c.metadata.get("chunk_index", "?")
            strategy = c.metadata.get("chunk_strategy", "")
            score    = c.score if c.score is not None else 0.0
            label    = f"[{i}]  {fname}  ·  {score:.4f}"
            with st.expander(label, expanded=(i == 1)):
                st.caption(
                    f"chunk {cidx}"
                    + (f"  ·  {strategy}" if strategy else "")
                )
                st.write(c.chunk_text)

    with answer_col:
        st.subheader("Answer")
        t_gen_start = time.perf_counter()
        try:
            # st.write_stream() accepts any str-yielding generator and renders
            # tokens as they arrive — no manual placeholder update loop needed.
            st.write_stream(
                pipeline._llm_chain.stream({"context": context, "question": q})
            )
            gen_s = time.perf_counter() - t_gen_start
        except Exception as exc:
            gen_s = time.perf_counter() - t_gen_start
            st.error(_generation_error_msg(exc))
            st.caption(
                "Retrieval succeeded — the source chunks on the right are still valid. "
                "Only the LLM generation step failed."
            )

        st.caption(
            f"Retrieved in {retrieval_s:.2f}s  ·  "
            f"Generated in {gen_s:.2f}s  ·  "
            f"{len(chunks)} chunks  ·  `{collection}`  ·  `{method}`"
        )


# ---------------------------------------------------------------------------
# Tab: Compare chunking strategies — helpers
# ---------------------------------------------------------------------------

def _eval_scores_for(
    collection: str,
    method: str,
    summary: dict,
) -> tuple[dict | None, str | None]:
    """Look up stored RAGAS scores for (collection, method).

    Returns (metrics_dict, variant_key_used) or (None, None) if not available.
    The variant key is included so the UI can show which evaluation run was used.
    """
    eval_variant = _EVAL_METHOD_MAP.get(method)
    if eval_variant is None:
        return None, None
    scores = summary.get(collection, {}).get(eval_variant, {}).get("metrics")
    return scores, eval_variant


def _agreement_note(results: dict) -> str:
    """Compare the top-1 retrieved chunk across all three collections.

    Returns a human-readable sentence that surfaces when chunking strategy
    produces genuinely different retrieval outcomes — the core thesis of the
    project made visible without the user having to compare chunks manually.
    """
    # Build (filename, chunk_index) pairs for each collection's top result
    top: dict[str, tuple[str, int] | None] = {}
    for col, res in results.items():
        if res["chunks"]:
            c = res["chunks"][0]
            top[col] = (
                c.metadata.get("filename", "?"),
                c.metadata.get("chunk_index", -1),
            )
        else:
            top[col] = None

    valid = {col: v for col, v in top.items() if v is not None}
    if not valid:
        return ""

    top_files = {col: v[0] for col, v in valid.items()}
    unique_files = set(top_files.values())

    if len(unique_files) == 1:
        fname = list(unique_files)[0]
        unique_pairs = set(valid.values())
        if len(unique_pairs) == 1:
            return f"All three strategies agree on the same top chunk — `{fname}`."
        return (
            f"All three strategies retrieved from `{fname}` "
            "but selected different chunks within it."
        )

    if len(unique_files) == 2:
        file_to_cols: dict[str, list[str]] = {}
        for col, f in top_files.items():
            file_to_cols.setdefault(f, []).append(col)
        majority_file  = max(file_to_cols, key=lambda f: len(file_to_cols[f]))
        minority_cols  = [c for c, f in top_files.items() if f != majority_file]
        minority_file  = top_files[minority_cols[0]]
        minority_label = " and ".join(COL_LABELS[c] for c in minority_cols)
        majority_label = " and ".join(COL_LABELS[c] for c in file_to_cols[majority_file])
        return (
            f"{minority_label} retrieved `{minority_file}` as the top source; "
            f"{majority_label} retrieved `{majority_file}`."
        )

    # All three differ
    file_list = ", ".join(
        f"`{top_files[c]}`" for c in _COMPARE_ORDER if c in top_files
    )
    return f"Each strategy retrieved a different top source: {file_list}."


def _compact_chunks(chunks: list, n: int = 3) -> None:
    """Render up to n chunks as compact previews.

    Shows a 150-character text preview inline; full chunk text is behind a
    'Show more' expander. Keeps columns readable without hiding the content.
    """
    for i, c in enumerate(chunks[:n], 1):
        fname   = c.metadata.get("filename", "unknown")
        cidx    = c.metadata.get("chunk_index", "?")
        score   = c.score if c.score is not None else 0.0
        preview = c.chunk_text[:150].replace("\n", " ").strip()
        clipped = len(c.chunk_text) > 150

        st.markdown(f"**[{i}]** `{fname}` · score {score:.3f}")
        st.caption(f"chunk {cidx}")
        st.write(preview + ("…" if clipped else ""))
        if clipped:
            with st.expander("Show more", expanded=False):
                st.write(c.chunk_text)
        if i < min(n, len(chunks)):
            st.divider()


# ---------------------------------------------------------------------------
# Tab: Compare chunking strategies
# ---------------------------------------------------------------------------

def tab_compare() -> None:
    st.header("Compare chunking strategies")
    st.caption(
        "Same question · same retrieval method · three different chunking strategies. "
        "Retrieval method and top-K are set in the sidebar."
    )

    question = st.text_area(
        "Question",
        placeholder="e.g. What are the three developmental paradigms of the RAG framework?",
        height=80,
        key="compare_question",
    )

    btn_col, _ = st.columns([1, 5])
    run = btn_col.button(
        "Compare strategies", type="primary", disabled=not question.strip()
    )

    if not question.strip() or not run:
        return

    method = st.session_state["method"]
    top_k  = st.session_state["top_k"]
    q      = question.strip()

    eval_data = _load_eval_data()
    summary   = eval_data[0] if eval_data else {}

    # ---- guard: warn about any empty collections before querying ------------
    empty_cols = [c for c in _COMPARE_ORDER if _collection_point_count(c) == 0]
    for c in empty_cols:
        st.warning(
            f"Collection `{c}` is empty — run `python -m src.indexing.index_documents` "
            "to populate it. Skipping this collection."
        )
    if len(empty_cols) == 3:
        return  # nothing to compare

    # ---- run retrieval + generation for all three collections ---------------
    results: dict = {}
    progress = st.progress(0, text="Querying collections…")
    for i, col in enumerate(_COMPARE_ORDER):
        if col in empty_cols:
            results[col] = {"chunks": [], "answer": "", "error": None, "skipped": True}
            continue
        progress.progress(i / 3, text=f"Querying {COL_LABELS[col]}…")
        pipeline = _get_pipeline(col)
        chunks   = _retrieve_with_top_k(pipeline, q, col, method, top_k)
        if not chunks:
            results[col] = {"chunks": [], "answer": "", "error": None, "skipped": False}
        else:
            context = format_context(chunks)
            try:
                answer = pipeline._llm_chain.invoke({"context": context, "question": q})
                results[col] = {"chunks": chunks, "answer": answer, "error": None, "skipped": False}
            except Exception as exc:
                # Store the exception object so _generation_error_msg can classify it
                results[col] = {"chunks": chunks, "answer": "", "error": exc, "skipped": False}
    progress.progress(1.0, text="Done.")
    progress.empty()

    # ---- cross-strategy retrieval comparison callout ------------------------
    note = _agreement_note(results)
    if note:
        st.info(f"🔍 **Retrieval comparison:** {note}")

    # ---- three-column results -----------------------------------------------
    col_widgets = st.columns(3)
    for col_widget, col_name in zip(col_widgets, _COMPARE_ORDER):
        res                  = results[col_name]
        scores, eval_variant = _eval_scores_for(col_name, method, summary)
        is_best              = col_name == "rag_recursive"

        with col_widget:
            label = COL_LABELS[col_name] + (" ⭐" if is_best else "")
            st.subheader(label)

            # ---- answer ----
            if res.get("skipped"):
                st.warning("Collection is empty — skipped.")
            elif res["error"]:
                st.error(_generation_error_msg(res["error"]))
                if res["chunks"]:
                    st.caption("Retrieval succeeded — sources below are still valid.")
            elif not res["answer"]:
                st.warning("No relevant context found for this collection.")
            else:
                st.markdown(res["answer"])

            st.divider()

            # ---- top-3 source chunks ----
            if res["chunks"]:
                st.caption(f"**Top sources** (3 of {len(res['chunks'])})")
                _compact_chunks(res["chunks"], n=3)
            else:
                st.caption("No chunks retrieved.")

            st.divider()

            # ---- stored RAGAS scores ----
            if scores:
                fa = scores.get("faithfulness", 0.0)
                ar = scores.get("answer_relevancy", 0.0)
                cp = scores.get("context_precision", 0.0)
                cr = scores.get("context_recall", 0.0)
                r1c1, r1c2 = st.columns(2)
                r1c1.metric("Faithfulness", f"{fa:.3f}")
                r1c2.metric("Ans. rel.",    f"{ar:.3f}")
                r2c1, r2c2 = st.columns(2)
                r2c1.metric("Ctx prec.",    f"{cp:.3f}")
                r2c2.metric("Ctx recall",   f"{cr:.3f}")
                st.caption(f"Eval scores · variant `{eval_variant}`")
            else:
                st.caption("No evaluation scores for this method.")


# ---------------------------------------------------------------------------
# Evaluation tab — chart helpers
# ---------------------------------------------------------------------------

_CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#fafafa",
)


def _render_findings() -> None:
    findings = Path("data/eval_findings.md")
    if not findings.exists():
        return
    with st.expander("📋 Findings & analysis (Day 6)", expanded=False):
        st.markdown(findings.read_text())


def _chart_metric_bars(df: pd.DataFrame, metric: str) -> None:
    """Grouped bar: collection on x, one bar per variant, y = selected metric."""
    fig = px.bar(
        df,
        x="collection",
        y=metric,
        color="variant",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={metric: METRIC_LABELS[metric], "collection": "Collection", "variant": "Variant"},
        height=400,
    )
    fig.update_layout(yaxis_range=[0, 1.08], legend_title="Variant", **_CHART_LAYOUT)
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)


def _chart_quality_latency(df: pd.DataFrame, metric: str) -> None:
    """Scatter: total latency (x) vs selected metric (y), colored by collection.

    Reveals whether the best-quality configuration is also the slowest —
    the upper-left quadrant is the ideal trade-off zone.
    """
    plot_df = df.dropna(subset=["total_latency_s", metric]).copy()
    plot_df["short_label"] = (
        plot_df["collection"].str.replace("rag_", "")
        + " / "
        + plot_df["variant"]
    )
    fig = px.scatter(
        plot_df,
        x="total_latency_s",
        y=metric,
        color="collection",
        symbol="variant",
        text="short_label",
        hover_data={
            "collection":       True,
            "variant":          True,
            metric:             ":.3f",
            "total_latency_s":  ":.2f",
            "short_label":      False,
        },
        labels={
            "total_latency_s": "Mean total latency / question (s)",
            metric:            METRIC_LABELS[metric],
        },
        height=400,
    )
    fig.update_traces(textposition="top center", marker_size=10)
    fig.update_layout(legend_title="Collection", **_CHART_LAYOUT)
    # Annotate the ideal quadrant
    fig.add_annotation(
        x=plot_df["total_latency_s"].min(),
        y=plot_df[metric].max(),
        text="← ideal (fast + high quality)",
        showarrow=False,
        font_color="#aaaaaa",
        xanchor="left",
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of all 4 RAGAS metrics across all 12 (collection, variant) runs."""
    metric_cols = list(METRIC_LABELS.keys())
    hm = df.copy()
    hm["run"] = hm["collection"] + " / " + hm["variant"]
    hm = hm.set_index("run")[metric_cols]

    fig = go.Figure(go.Heatmap(
        z=hm.values,
        x=[METRIC_LABELS[m] for m in metric_cols],
        y=hm.index.tolist(),
        colorscale="RdYlGn",
        zmin=0.5, zmax=1.0,
        text=[[f"{v:.3f}" for v in row] for row in hm.values],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="RAGAS scores — all 12 runs",
        height=520,
        margin=dict(l=290, r=40, t=60, b=40),
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_per_question(pq_df: pd.DataFrame | None) -> None:
    """Sortable, filterable per-question score table.

    Lets you drill into specific questions where one strategy clearly
    outperformed another — the 'why' behind the aggregate numbers.
    """
    st.subheader("Per-question breakdown")
    if pq_df is None:
        st.caption("No per-question CSVs found in `data/compare/`.")
        return

    fc1, fc2 = st.columns(2)
    with fc1:
        sel_cols = st.multiselect(
            "Collections",
            options=["rag_fixed", "rag_recursive", "rag_semantic"],
            default=["rag_recursive"],
            key="pq_cols",
        )
    with fc2:
        sel_vars = st.multiselect(
            "Variants",
            options=sorted(pq_df["variant"].unique()),
            default=["hybrid_rrf+rerank"],
            key="pq_vars",
        )

    if not sel_cols or not sel_vars:
        st.caption("Select at least one collection and one variant.")
        return

    filtered = pq_df[
        pq_df["collection"].isin(sel_cols) & pq_df["variant"].isin(sel_vars)
    ].copy()

    if filtered.empty:
        st.caption("No rows match the selected filters.")
        return

    want = [
        "id", "question", "collection", "variant",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "top_retrieved_doc", "source_document",
    ]
    show_cols = [c for c in want if c in filtered.columns]
    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                   if c in show_cols]

    st.caption(f"{len(filtered)} question-run pairs · sorted by faithfulness ↓ · click any column header to re-sort")
    st.dataframe(
        filtered[show_cols]
        .sort_values("faithfulness", ascending=False)
        .reset_index(drop=True)
        .style.format({c: "{:.3f}" for c in metric_cols})
        .background_gradient(subset=metric_cols, cmap="RdYlGn", vmin=0.0, vmax=1.0),
        use_container_width=True,
        height=480,
    )


# ---------------------------------------------------------------------------
# Tab: Evaluation Results
# ---------------------------------------------------------------------------

def tab_evaluation() -> None:
    st.header("Evaluation results")
    st.caption("3 collections × 4 retrieval variants × 27 questions = 12 MLflow runs.")

    # ---- load data ----------------------------------------------------------
    runs_df       = _load_mlflow_runs()
    eval_fallback = _load_eval_data()
    pq_df         = _load_per_question_df()

    if runs_df is not None and len(runs_df) >= 12:
        chart_df = runs_df
    elif eval_fallback is not None:
        _, chart_df = eval_fallback
        note = (
            f"MLflow returned {len(runs_df)} runs (expected 12)."
            if runs_df is not None else "MLflow data unavailable."
        )
        st.caption(f"{note} Showing comparison_summary.json instead.")
    else:
        st.error(
            f"No evaluation data found at `{SUMMARY_PATH}`. "
            "Run `python -m src.evaluation.compare` first."
        )
        return

    metric_cols = list(METRIC_LABELS.keys())

    # ---- findings -----------------------------------------------------------
    _render_findings()
    st.divider()

    # ---- metric selector (shared for charts 1 & 2) -------------------------
    selected_metric = st.selectbox(
        "Metric to explore",
        options=metric_cols,
        format_func=lambda k: METRIC_LABELS[k],
    )

    # ---- chart 1: grouped bar + chart 2: scatter (side by side) ------------
    bar_col, scatter_col = st.columns(2)
    with bar_col:
        st.subheader(f"{METRIC_LABELS[selected_metric]} by configuration")
        _chart_metric_bars(chart_df, selected_metric)
    with scatter_col:
        st.subheader(f"Quality vs latency")
        _chart_quality_latency(chart_df, selected_metric)

    # ---- chart 3: heatmap --------------------------------------------------
    st.divider()
    _chart_heatmap(chart_df)

    # ---- chart 4: per-question table ----------------------------------------
    st.divider()
    _chart_per_question(pq_df)

    # ---- raw numbers -------------------------------------------------------
    with st.expander("Raw aggregate numbers", expanded=False):
        display_cols = ["collection", "variant"] + metric_cols + ["retrieval_latency_s", "generation_latency_s"]
        display_cols = [c for c in display_cols if c in chart_df.columns]
        st.dataframe(
            chart_df[display_cols]
            .rename(columns={
                **METRIC_LABELS,
                "retrieval_latency_s":  "Retrieval (s)",
                "generation_latency_s": "Generation (s)",
            })
            .style.format({
                **{METRIC_LABELS[m]: "{:.4f}" for m in metric_cols},
                "Retrieval (s)":  "{:.3f}",
                "Generation (s)": "{:.3f}",
            })
            .highlight_max(
                subset=[METRIC_LABELS[m] for m in metric_cols],
                color="rgba(0,200,120,0.3)",
            ),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> None:
    with st.sidebar:
        st.title("📚 doc-rag-engine")
        st.caption("Hybrid search · cross-encoder reranking · RAGAS evaluation")
        st.divider()

        # ---- Query settings (shared across Ask and Compare tabs) -----------
        st.subheader("Query settings")

        # key= binds the widget to st.session_state automatically.
        # Changing the value here persists when the user switches tabs.
        st.selectbox(
            "Collection",
            options=COLLECTION_NAMES,
            format_func=lambda k: COLLECTION_LABELS[k],
            key="collection",
            help="The Qdrant collection to search. rag_recursive scored highest on faithfulness and context recall.",
        )

        st.selectbox(
            "Retrieval method",
            options=list(METHOD_LABELS.keys()),
            format_func=lambda k: METHOD_LABELS[k],
            key="method",
            help="First-stage retrieval strategy. The cross-encoder reranker always runs afterwards.",
        )

        st.slider(
            "Chunks passed to LLM (top-K)",
            min_value=1,
            max_value=20,
            key="top_k",
            help=(
                "The cross-encoder scores all 20 first-stage candidates. "
                "This slider controls how many top-scored chunks are included "
                "in the LLM prompt. Higher K = more context, slower generation."
            ),
        )

        st.divider()

        # ---- Active configuration (read from cached AppConfig) -------------
        st.subheader("Active configuration")
        _, _, cfg = _get_shared()
        st.markdown(f"**Embedding model**  \n`{cfg.dense_embedding_model}`")
        st.markdown(f"**LLM**  \n`{cfg.ollama_model}` via `{cfg.ollama_base_url}`")
        st.markdown(f"**Rerank model**  \n`{cfg.rerank_model}`")
        st.markdown(f"**Qdrant**  \n`{cfg.qdrant_host}:{cfg.qdrant_port}`")
        st.markdown(f"**First-stage candidates**  \n`{cfg.retrieval_limit}`")

        st.divider()

        # ---- Best config callout -------------------------------------------
        st.caption(
            "Best RAGAS config: `rag_recursive + hybrid_rrf`  \n"
            "Faithfulness **0.802** · Context recall **0.944**"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _system_overview() -> None:
    with st.expander("How this system works", expanded=False):
        st.markdown(
            "An end-to-end RAG pipeline that indexes documents into Qdrant, "
            "retrieves with hybrid dense + BM25 search, reranks with a cross-encoder, "
            "generates answers via an OpenAI-compatible LLM (Ollama), "
            "and evaluates retrieval + generation quality with RAGAS across all configuration combinations."
        )
        st.code(
            """\
Raw Documents (PDF, DOCX, PPTX, HTML, MD, CSV, XLSX, TXT)
    |
    v  src/ingestion/
       DocumentLoader routes plain text directly; all other formats through docling.
    |
    v  src/chunking/       [three parallel strategies, one Qdrant collection each]
         FixedSizeChunker  — fixed character window with overlap
         RecursiveChunker  — splits on paragraph / sentence / word boundaries
         SemanticChunker   — groups semantically similar sentences using embeddings
    |
    v  src/embedding/
         Dense vectors  — MiniLM all-MiniLM-L6-v2 (384-dim)
         Sparse vectors — SPLADE prithivida/Splade_PP_en_v1
    |
    v  src/vectorstore/    [Qdrant: rag_fixed · rag_recursive · rag_semantic]
    |
    v  src/retrieval/                                          [query time]
         DenseRetriever   — cosine similarity over 384-dim vectors
         BM25Retriever    — Okapi BM25 in-memory index over all chunks
         HybridRetriever  — RRF fusion of dense + BM25, fetch top 20 candidates
         Reranker         — FlashRank cross-encoder, rerank top 20 → top K
    |
    v  src/generation/
         RAGPipeline.invoke() — LCEL: format context → DETAILED_PROMPT → LLM → parse
         RAGPipeline.stream() — token-streaming variant used by this dashboard
    |
    v  src/evaluation/
         compare.py — 3 collections × 4 variants = 12 MLflow runs
         RAGAS metrics: faithfulness · answer_relevancy · context_precision · context_recall""",
            language="text",
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Collections", "3")
        c2.metric("Documents indexed", "14")
        c3.metric("Qdrant points", "39,214")
        c4.metric("Eval questions", "27")


def main() -> None:
    _sidebar()
    _system_overview()

    if not _qdrant_reachable():
        st.error(
            "**Qdrant is not reachable.**  \n"
            "The vector store container is not running. Start it with:"
        )
        st.code("docker compose up -d", language="bash")
        st.caption("Once the container is healthy, refresh this page.")
        st.stop()

    ask_tab, compare_tab, eval_tab = st.tabs(["Ask", "Compare Collections", "Evaluation Results"])

    with ask_tab:
        tab_ask()

    with compare_tab:
        tab_compare()

    with eval_tab:
        tab_evaluation()


if __name__ == "__main__":
    main()

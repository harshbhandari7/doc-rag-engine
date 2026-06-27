# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An advanced RAG pipeline with hybrid search (dense + sparse), three chunking strategies (fixed-size, recursive, semantic), LLM answer generation, and a RAGAS-based evaluation framework. The pipeline indexes documents into Qdrant, retrieves with RRF fusion, reranks with a cross-encoder, generates answers via any OpenAI-compatible LLM API, and evaluates retrieval + generation quality across all configuration combinations, logging results to MLflow.

Sample documents are in `data/raw/` (ArXiv ML papers, Gutenberg classics, economic reports) — gitignored, not committed.

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Start Qdrant (required for indexing and retrieval):

```bash
docker compose up -d
```

Environment variables in `.env` (gitignored). Active values:

```
APP_CHUNK_SIZE=512
APP_CHUNK_OVERLAP=64
APP_SEMANTIC_BREAKPOINT_THRESHOLD=95
APP_DENSE_EMBEDDING_MODEL=all-MiniLM-L6-v2
APP_DENSE_EMBEDDING_DIM=384
APP_SPARSE_EMBEDDING_MODEL=prithivida/Splade_PP_en_v1
APP_QDRANT_HOST=localhost
APP_QDRANT_PORT=6333
APP_COLLECTION_FIXED=rag_fixed
APP_COLLECTION_RECURSIVE=rag_recursive
APP_COLLECTION_SEMANTIC=rag_semantic

OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=<your_key>
OLLAMA_MODEL=gemma3:4b
OLLAMA_TEMPERATURE=0.1
OLLAMA_MAX_TOKENS=1024
REQUEST_TIMEOUT=60

RAGAS_JUDGE_MODEL=gemma3:12b
RAGAS_MAX_WORKERS=2

MLFLOW_TRACKING_URI=sqlite:///mlruns.db
MLFLOW_EXPERIMENT_NAME=rag_chunking_comparison
```

## Architecture

```
Raw Documents (PDF, DOCX, PPTX, HTML, MD, CSV, XLSX, TXT)
    ↓ src/ingestion/      — DocumentLoader → LoadedDocument (docling + plain-text fallback)
    ↓ src/chunking/       — FixedSizeChunker / RecursiveChunker / SemanticChunker → ChunkedDocument
    ↓ src/embedding/      — Embedder.embed_dense() (MiniLM) + embed_sparse() (SPLADE)
    ↓ src/vectorstore/    — VectorStore: Qdrant collections with named dense+sparse vectors, hybrid RRF search
    ↓ src/indexing/       — index_documents.py: one-time script to populate all three collections
    ↓ src/retrieval/      — DenseRetriever / BM25Retriever / HybridRetriever / Reranker → RetrievalResult
    ↓ src/generation/     — RAGPipeline: LCEL chain (prompt | ChatOpenAI | StrOutputParser), invoke + stream
    ↓ src/evaluation/     — compare.py: 3 collections × 4 variants = 12 MLflow runs via RAGAS
```

## Module Reference

### `src/models.py`
- `RetrievalResult` — unified output dataclass for all retrieval stages
- Fields: `chunk_text`, `score`, `metadata` (dict with `chunk_strategy`, `chunk_index`, `chunk_size`, `parent_source`, `filename`, `format`), `retrieval_method`
- All retrievers and the reranker return this type; downstream code never needs to know which method produced a result

### `src/ingestion/`
- `loader.py` — `DocumentLoader`: routes plain-text directly, all other formats through docling; `load_batch` uses docling's `convert_all` for concurrent processing; applies empty-page filtering then small-page merging
- `models.py` — `LoadedDocument`, `PageContent`, `DocumentMetadata`, `LoadStatus`

```python
from src.ingestion import DocumentLoader
loader = DocumentLoader()
doc = loader.load("data/raw/arxiv_papers/attention_need.pdf")
for doc in loader.load_directory("data/raw"):
    print(doc.metadata.filename, doc.status)
```

### `src/chunking/`
- `chunkers.py` — `FixedSizeChunker`, `RecursiveChunker`, `SemanticChunker`, `ChunkedDocument`
- All take `list[LoadedDocument]` and return `list[ChunkedDocument]`
- `SemanticChunker` operates on full concatenated document text (not page-by-page) and caps oversized chunks via recursive fallback splitter
- `ChunkedDocument` fields: `text`, `metadata` (original), `chunk_strategy`, `chunk_index`, `chunk_size`, `parent_source`

```python
from src.chunking import FixedSizeChunker, RecursiveChunker, SemanticChunker
chunks = RecursiveChunker().chunk([doc])
```

### `src/embedding/`
- `embedder.py` — `Embedder`, `SparseEmbedding`
- `embed_dense(texts)` → `np.ndarray` shape `(n, 384)` float32
- `embed_sparse(texts)` → `list[SparseEmbedding]` (indices + values; SPLADE lazy-loads on first call, logs ~500MB download warning)
- Dense batch_size=32, sparse batch_size=16 (SPLADE is slower)

```python
from src.embedding import Embedder
embedder = Embedder()
dense  = embedder.embed_dense(["chunk text"])   # (1, 384)
sparse = embedder.embed_sparse(["chunk text"])  # [SparseEmbedding]
```

### `src/vectorstore/`
- `store.py` — `VectorStore`, `SearchResult`
- Abstraction boundary: nothing outside this file imports from `qdrant_client`
- `ensure_collection(name, recreate=False)` — idempotent; creates dense + sparse named vector spaces
- `upsert(chunks, dense_vectors, sparse_vectors, collection, batch_size=100)` — deterministic UUID5 point IDs
- `search_dense(query_vector, collection, limit, filters)` — cosine similarity
- `search_sparse(query_sparse, collection, limit)` — SPLADE keyword search
- `search_hybrid(query_dense, query_sparse, collection, limit, filters)` — prefetch top-20 from each, RRF fusion
- `scroll_all(collection, batch=1000)` → `list[dict]` — returns all point payloads; used by `BM25Retriever` to build its in-memory index
- `filters` accepts plain `dict` e.g. `{"source_file": "report.pdf"}` — converted to Qdrant Filter internally

```python
from src.vectorstore import VectorStore, SearchResult
store = VectorStore()
store.ensure_collection("rag_fixed", recreate=True)
store.upsert(chunks, dense, sparse, "rag_fixed")
hits = store.search_hybrid(q_dense, q_sparse, "rag_fixed", limit=5)
```

### `src/indexing/`
- `index_documents.py` — one-time script; loads all docs, runs all three chunkers, embeds, upserts
- Run from project root: `python -m src.indexing.index_documents`
- Flags: `--data-dir data/raw`, `--no-recreate`
- Already run: 14 docs → rag_fixed (10,439 pts), rag_recursive (12,389 pts), rag_semantic (16,386 pts)

### `src/retrieval/`

The query-time retrieval pipeline. All classes operate on `RetrievalResult` from `src/models.py`. Inject shared `Embedder` and `VectorStore` instances — do not instantiate per-retriever.

- `dense.py` — `DenseRetriever(embedder, store)`
  - `retrieve(query, collection, limit=20, filters=None)` → `list[RetrievalResult]` tagged `"dense"`

- `sparse.py` — `BM25Retriever(store, collection)` and `SparseRetriever(embedder, store)`
  - `BM25Retriever` builds an in-memory Okapi BM25 index at construction (~10s for 12k chunks); `retrieve(query, limit=20)` — no per-call collection arg (fixed at construction)
  - `SparseRetriever` embeds with SPLADE, queries Qdrant sparse index; `retrieve(query, collection, limit=20)`

- `hybrid.py` — `HybridRetriever(dense, sparse, config=None)`
  - `retrieve_rrf(query, collection, limit=20, k=None)` → fused list tagged `"hybrid_rrf"`
  - `retrieve_weighted(query, collection, limit=20, alpha=None)` → fused list tagged `"hybrid_weighted"`

- `reranker.py` — `Reranker(config=None)`
  - `rerank(query, results, top_n=None)` → reranked `list[RetrievalResult]` tagged `"reranked"`
  - FlashRank cross-encoder (`ms-marco-MiniLM-L-12-v2`, ~90MB, downloads to `/tmp` on first use)

```python
from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever, Reranker

dense    = DenseRetriever(embedder, store)
bm25     = BM25Retriever(store, cfg.collection_recursive)
hybrid   = HybridRetriever(dense, bm25, cfg)
reranker = Reranker(cfg)

candidates = hybrid.retrieve_rrf(query, collection, limit=20)
top5 = reranker.rerank(query, candidates, top_n=5)
```

**Known failure mode — RRF false-positive:** When a borderline document ranks 2nd in BM25 and 4th in dense, its combined RRF score can beat the correct answer. Observed empirically: hybrid promoted a wrong document to rank 1 while both individual methods were correct. Always run the reranker — it corrects this.

### `src/generation/`

- `pipeline.py` — `RAGPipeline(embedder, store, collection, config=None, use_detailed_prompt=True)`
  - Construct once (BM25 index + cross-encoder load happen here). Call `invoke`/`stream` many times.
  - `use_detailed_prompt=True` (default) uses `DETAILED_PROMPT` with explicit "do not use outside knowledge" grounding. Evaluation scripts pass `False` to reduce token usage.
  - `invoke(question, collection=None, retrieval_method="hybrid_rrf")` → always returns:
    ```python
    {
        "answer":           str,
        "source_chunks":    list[RetrievalResult],
        "retrieval_method": str,
        "collection":       str,
        "query":            str,
        "error":            str | None,
    }
    ```
  - `stream(question, collection=None, retrieval_method="hybrid_rrf")` → `Iterator[str]`
  - `_retrieve(question, collection, retrieval_method, rerank=True)` — internal; `rerank=False` used by `compare.py` to isolate cross-encoder contribution
  - Three error guards: empty retrieval (early return), LLM timeout (try/except), context too long (`_truncate_to_limit` drops tail chunks)
  - Uses `ChatOpenAI` pointed at Ollama's `/v1/` endpoint — works for both cloud and local Ollama

- `prompts.py` — `CONCISE_PROMPT`, `DETAILED_PROMPT`, `format_context(results)`
  - `format_context` produces numbered `[1] Source: filename | Chunk N` headers
  - `CONCISE_PROMPT`: system + human messages, `input_variables=['context','question']`

```python
from src.generation import RAGPipeline

pipeline = RAGPipeline(embedder, store, cfg.default_collection)
result   = pipeline.invoke("What is the RAGAS faithfulness metric?")
print(result["answer"])

for token in pipeline.stream("How does multi-head attention work?"):
    print(token, end="", flush=True)
```

### `src/evaluation/`

- `judge.py` — `make_judge(config=None)` → `(LangchainLLMWrapper, LangchainEmbeddingsWrapper)`
  - Builds RAGAS judge using `ChatOpenAI` on Ollama's `/v1/` endpoint + `HuggingFaceEmbeddings`
  - Construct once per evaluation run, pass to `ragas_evaluate()`

- `ragas_eval.py`
  - `make_metrics()` → `[faithfulness, answer_relevancy, context_precision, context_recall]` (old singleton instances from `ragas.metrics._*` — required by `ragas_evaluate()` in ragas 0.4.x)
  - `to_ragas_dataset(entries)` → `EvaluationDataset` — maps `question→user_input`, `answer→response`, `contexts→retrieved_contexts`, `ground_truth→reference`
  - `run_ragas(eval_dataset, judge_llm, judge_emb, raise_exceptions, max_workers)` → `(means_dict, EvaluationResult)`
  - **ragas 0.4.x API notes:** use old singleton metrics (NOT `ragas.metrics.collections`); pass `llm=` and `embeddings=` to `ragas_evaluate()`; always pass `RunConfig(max_workers=2)` to avoid hammering free-tier rate limits

- `compare.py` — full 3×4 comparison matrix
  - `Variant(label, method, rerank)` — four variants: `dense`, `bm25`, `hybrid_rrf`, `hybrid_rrf+rerank`
  - One `RAGPipeline` per collection (BM25 index builds once, reused for all 4 variants)
  - Separate `time.perf_counter()` around `_retrieve()` and `_llm_chain.invoke()` for independent latency tracking
  - Saves raw JSON before RAGAS — use `--skip-collect` to re-run only the judge after a crash
  - CLI: `--dry-run`, `--skip-collect`, `--collections`, `--variants`

```bash
python -m src.evaluation.compare                          # full 12-run matrix
python -m src.evaluation.compare --skip-collect           # re-run RAGAS only
python -m src.evaluation.compare --collections rag_recursive --variants hybrid_rrf "hybrid_rrf+rerank"
```

- `build_dataset.py` — standalone script: `python -m src.evaluation.build_dataset`

## Live Qdrant State

All three collections are populated and `green`. Hybrid search verified working.

| Collection | Strategy | Points |
|---|---|---|
| `rag_fixed` | Fixed-size | 10,439 |
| `rag_recursive` | Recursive | 12,389 |
| `rag_semantic` | Semantic | 16,386 |

Each point has: dense vector (384-dim cosine) + sparse vector (SPLADE) + payload with `chunk_text`, `source_file`, `chunk_strategy`, `chunk_index`, `chunk_size`, `parent_source`, `filename`, `format`, `total_pages`.

## Evaluation Data

- `data/test_questions.json` — 27 questions across 14 documents with verbatim ground truths
- `data/eval_dataset.json` — collected pipeline outputs (contexts fully populated; answers require LLM)
- `data/compare/` — per-run raw JSON + CSV artifacts from `compare.py`
- `mlruns.db` — MLflow SQLite backend; launch UI with `mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5001`
  - **Note:** macOS reserves port 5000 for AirPlay Receiver — always use `--port 5001` or higher

## Dashboard

`app.py` — Streamlit app, run with `streamlit run app.py` (port 8501).

Three tabs:
- **Ask** — streaming single query; example question buttons pre-fill via `on_click` callback; two-column layout (answer streams left, sources populated right); retrieval + generation latency caption below answer
- **Compare Collections** — same question → all 3 collections in parallel columns; compact 150-char chunk previews; stored RAGAS scores per column; cross-strategy top-1 agreement callout
- **Evaluation Results** — charts from MLflow (primary) or `data/compare/comparison_summary.json` (fallback); grouped bar, quality-vs-latency scatter, full heatmap, per-question sortable table

### Caching strategy

Streamlit reruns the entire script on every user interaction. Three cache layers prevent model reloads:

- `@st.cache_resource` — non-serialisable objects; shared across all sessions; no TTL
  - `_get_shared()` → `(Embedder, VectorStore, AppConfig)` — constructed once per process
  - `_get_pipeline(collection)` → one `RAGPipeline` per collection (BM25 index + cross-encoder each load here)
- `@st.cache_data(ttl=30)` — `_qdrant_reachable()` pings Qdrant; auto-recovers within 30s of Docker starting
- `@st.cache_data(ttl=60)` — `_collection_point_count(collection)` checks for empty collections before building pipelines
- `@st.cache_data` (no TTL) — `_load_eval_data()`, `_load_mlflow_runs()`, `_load_per_question_df()` for evaluation data (disk reads that don't change between `compare.py` runs)

Never cached: `pipeline.stream()` / `pipeline.invoke()` calls — every question is a live query.

### Session state

```python
_SS_DEFAULTS = {
    "collection": "rag_recursive",
    "method":     "hybrid_rrf",
    "top_k":      5,
    "query_text": "",
}
```

All selector widgets use `key=` to bind to `st.session_state` — values persist when the user switches tabs. The `if _k not in st.session_state` guard at module level sets defaults only once per server process.

### Key helpers

- `_retrieve_with_top_k(pipeline, question, collection, method, top_k)` — calls individual retrievers directly to get 20 first-stage candidates, then reranks to `top_k`. Bypasses `pipeline._retrieve()` so the sidebar slider can override chunk count without mutating shared `AppConfig`.
- `_generation_error_msg(exc)` — classifies LLM exceptions (401, 429, connection refused, timeout) into user-readable guidance strings.
- `_agreement_note(results)` — compares `(filename, chunk_index)` of top-1 result across all three collections; returns sentence describing where strategies agree or diverge.
- `_compact_chunks(chunks, n=3)` — renders 150-char inline previews with "Show more" expander; keeps compare columns readable.
- `_eval_scores_for(collection, method, summary)` — maps sidebar method key → evaluation variant via `_EVAL_METHOD_MAP` to look up stored RAGAS scores.

### Streaming

`st.write_stream(pipeline._llm_chain.stream({"context": context, "question": q}))` — Streamlit's native streaming delta protocol. No manual placeholder update loop. Sources column is populated before streaming starts so the user reads sources while the answer generates.

### MLflow data loading

`_load_mlflow_runs()` uses `mlflow.get_experiment_by_name()` → `search_runs(experiment_ids=[...])` — MLflow 3.x requires IDs, not names. Deduplicates by keeping the most recent run per `(collection, variant)` (re-runs produce duplicate rows). Falls back to `comparison_summary.json` if <12 runs found.

`_load_per_question_df()` globs `data/compare/scores_*.csv`; parses collection and variant from filename pattern `scores_{collection}_{variant}.csv` since the CSVs themselves have no collection/variant column.

### Qdrant gate

`main()` checks `_qdrant_reachable()` before rendering any tab. On failure: error message + `docker compose up -d` code block + `st.stop()`. Health check re-runs every 30s, so the page auto-recovers on refresh after Docker starts.

### All dashboard pipelines use `use_detailed_prompt=True`

Enforces explicit "do not use outside knowledge" grounding. Out-of-scope questions return the sentinel phrase from `DETAILED_PROMPT` rather than hallucinated answers.

## Notebooks

- `notebooks/splitter_comparison.ipynb` — compares fixed/recursive/semantic chunking output
- `notebooks/retrieval_sanity_check.ipynb` — same query across all three collections, verifies search types and metadata filtering
- `notebooks/retrieval_pipeline_eval.ipynb` — end-to-end pipeline evaluation: runs 3 test questions through dense / BM25 / hybrid RRF / reranker
- `notebooks/generation_pipeline_test.ipynb` — full RAG pipeline test: 3 questions, cross-collection comparison, streaming demo, error guard verification

## Key Libraries

| Layer | Library |
|---|---|
| Document parsing | `docling`, `pypdfium2`, `rapidocr`, `ocrmac` |
| Office formats | `python-docx`, `python-pptx`, `openpyxl` |
| Chunking | `langchain-text-splitters` |
| Dense embeddings | `sentence-transformers`, `torch`, `transformers` |
| Sparse embeddings | `fastembed` (ONNX runtime) |
| Vector store | `qdrant-client` |
| Sparse retrieval | `rank-bm25` (Okapi BM25 in-memory index) |
| Reranking | `flashrank` (FlashRank cross-encoder) |
| Generation | `langchain-openai`, `langchain-core` (LCEL chain) |
| Evaluation | `ragas` 0.4.x, `mlflow` (SQLite backend) |
| Dashboard | `streamlit`, `plotly` |
| Configuration | `pydantic-settings`, `python-dotenv` |
| Testing / DX | `pytest` 8.x, `ruff` 0.9+ |

## Configuration

Single `AppConfig` in `configs/settings.py` with `APP_` env prefix. Fields using `validation_alias` bypass the prefix and read their own env var name.

- **Ingestion:** `APP_OCR_ENABLED`, `APP_TABLE_STRUCTURE_ENABLED`, `APP_DOCUMENT_TIMEOUT`, `APP_MIN_PAGE_CHARS`, `APP_MERGE_MIN_CHARS`
- **Chunking:** `APP_CHUNK_SIZE`, `APP_CHUNK_OVERLAP`, `APP_SEMANTIC_BREAKPOINT_THRESHOLD`
- **Embedding:** `APP_DENSE_EMBEDDING_MODEL`, `APP_DENSE_EMBEDDING_DIM`, `APP_SPARSE_EMBEDDING_MODEL`
- **Hybrid retrieval:** `APP_RRF_K` (default 60), `APP_HYBRID_ALPHA` (default 0.7)
- **Reranking:** `APP_RERANK_MODEL`, `APP_RETRIEVAL_LIMIT` (default 20), `APP_RERANK_TOP_N` (default 5)
- **Generation:** `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`, `OLLAMA_TEMPERATURE`, `OLLAMA_MAX_TOKENS`, `REQUEST_TIMEOUT`, `APP_MAX_CONTEXT_CHARS` (default 12000), `APP_DEFAULT_COLLECTION` (default `rag_recursive`), `APP_DEFAULT_RETRIEVAL_METHOD` (default `hybrid_rrf+rerank` — evaluation-backed)
- **Evaluation:** `RAGAS_JUDGE_MODEL`, `RAGAS_MAX_WORKERS` (default 2 — keep low for free-tier APIs), `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`
- **Vector store:** `APP_QDRANT_HOST`, `APP_QDRANT_PORT`, `APP_COLLECTION_FIXED`, `APP_COLLECTION_RECURSIVE`, `APP_COLLECTION_SEMANTIC`

## Testing

Unit tests cover the retrieval logic, chunking helpers, and prompts — no external services required.

```bash
pip install -r requirements-dev.txt   # pytest, ruff (one-time)
pytest tests/unit/                    # 76 tests, ~0.5s
pytest tests/ -m integration -v      # real Qdrant required (docker compose up -d)
```

Test layout:
- `tests/unit/test_hybrid.py` — `reciprocal_rank_fusion` and `weighted_score_fusion` pure functions (12 + 11 tests); highest-value tests since bugs here are silent at the pipeline level
- `tests/unit/test_bm25.py` — `BM25Retriever` with mocked `VectorStore.scroll_all`; tests scoring, filtering, metadata mapping
- `tests/unit/test_dense.py` — `DenseRetriever` with mocked `Embedder` and `VectorStore`; tests vector handoff, metadata mapping
- `tests/unit/test_chunkers.py` — `FixedSizeChunker`, `RecursiveChunker`, and private helpers `_split_sentences`, `_cosine_distances`, `_make_chunks`
- `tests/unit/test_prompts.py` — `format_context` output format, numbering, fallbacks
- `tests/unit/test_models.py` — `RetrievalResult` dataclass field access
- `tests/integration/test_retrieval.py` — full retrieval stack against live Qdrant + populated collections; skipped by default

**Known BM25 quirk documented in tests:** `BM25Okapi` IDF = `log(N-df+0.5) - log(df+0.5)` evaluates to 0 when N=2 and df=1 (any term in exactly half the corpus). Tests use N≥3 documents so unique terms get positive IDF. If you're writing new BM25 tests, include at least 3 corpus documents.

## Development Notes

- Python 3.13 (`venv/` gitignored)
- `pyproject.toml` — pytest config (`testpaths`, `pythonpath=["."]`, `integration` marker) + ruff lint config (`select=["E","F","I"]`)
- `requirements-dev.txt` — dev-only deps (`pytest>=8.3`, `ruff>=0.9`); not needed at runtime
- `.env.example` — committed template with all env vars, annotated; copy to `.env` and fill in secrets
- `.env` gitignored; place secrets and local overrides there
- `data/raw/.gitkeep` — placeholder so the `data/raw/` directory exists after cloning (all other contents of `data/raw/` are gitignored)
- `data/raw/` gitignored — PDFs are large binaries
- `data/compare/` gitignored — generated evaluation artifacts, can be reproduced
- `mlruns.db` gitignored — MLflow SQLite backend, reproduced by re-running compare.py
- SPLADE model downloads to `~/.cache/` on first `embed_sparse()` call (~500MB, one-time)
- FlashRank cross-encoder downloads to `/tmp` on first `Reranker()` construction (~90MB, one-time)
- OCR is enabled by default (`APP_OCR_ENABLED=true`) — set to `false` for digital PDFs to avoid slow processing
- Qdrant dashboard available at http://localhost:6333/dashboard when container is running
- `ChatOllama` does NOT work with Ollama cloud (no auth header support) — use `ChatOpenAI` with `base_url=OLLAMA_BASE_URL/v1/` instead
- ragas 0.4.x: use old singleton metrics from `ragas.metrics._*`, NOT `ragas.metrics.collections`; the collections API requires `InstructorBaseRagasLLM` which is incompatible with `ragas_evaluate()`
- RAGAS judge fires 16 concurrent requests by default — set `RAGAS_MAX_WORKERS=2` for free-tier APIs to avoid 429s

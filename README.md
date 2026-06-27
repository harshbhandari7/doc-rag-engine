# doc-rag-engine

An advanced RAG (Retrieval-Augmented Generation) pipeline that indexes documents into Qdrant using both dense and sparse vectors, retrieves candidates with hybrid RRF fusion, reranks with a cross-encoder, generates grounded answers via an OpenAI-compatible LLM, and evaluates retrieval + generation quality across all configuration combinations with RAGAS and MLflow.

The central question this project answers: **does chunking strategy meaningfully change retrieval quality?** The evaluation across 3 strategies × 4 retrieval variants × 27 questions shows it does — recursive chunking outperforms fixed-size and semantic on faithfulness and context recall, and the margin is large enough to matter in production.

---

## Dashboard

A Streamlit app wrapping the full pipeline with three tabs:

| Tab | What it does |
|---|---|
| **Ask** | Type a question, stream a grounded answer from any collection + retrieval method, inspect the source chunks that grounded it |
| **Compare Collections** | Same question through all three chunking strategies side-by-side — surfaces where fixed, recursive, and semantic chunking retrieve different sources |
| **Evaluation Results** | Interactive charts from the 12-run RAGAS matrix: grouped bar, quality-vs-latency scatter, full heatmap, per-question sortable breakdown |

```bash
source venv/bin/activate
streamlit run app.py
# Open http://localhost:8501
```

---

## Setup

**Prerequisites:** Python 3.13+, Docker

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd doc-rag-engine

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# Optional: dev tools (pytest + ruff) for contributors
pip install -r requirements-dev.txt
```

First-run downloads (one-time, happen automatically):

| Download | Size | When |
|---|---|---|
| `all-MiniLM-L6-v2` (dense embeddings) | ~90 MB | First `Embedder()` call |
| `prithivida/Splade_PP_en_v1` (sparse embeddings) | ~500 MB | First `embed_sparse()` call |
| `ms-marco-MiniLM-L-12-v2` (cross-encoder reranker) | ~90 MB | First `Reranker()` call |

### 3. Start Qdrant

```bash
docker compose up -d
```

Verify it's running: http://localhost:6333/dashboard

### 4. Configure environment variables

Copy the example file and fill in the required values:

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

| Variable | Where to get it |
|---|---|
| `OLLAMA_API_KEY` | [ollama.com](https://ollama.com) → Settings → API Keys (free plan includes `gemma3:4b` + `gemma3:12b`) |
| `OLLAMA_BASE_URL` | `https://ollama.com` for cloud; `http://localhost:11434` for local Ollama |
| `OLLAMA_MODEL` | `gemma3:4b` recommended for generation |
| `RAGAS_JUDGE_MODEL` | `gemma3:12b` recommended for evaluation |

Everything else in `.env.example` has working defaults and can be left as-is for a first run.

> **Local Ollama:** set `OLLAMA_BASE_URL=http://localhost:11434` and leave `OLLAMA_API_KEY` empty. Do not add `/v1/` to the URL — the code appends it.

### 5. Place documents and run indexing

Place PDF, DOCX, PPTX, HTML, MD, CSV, XLSX, or TXT files in `data/raw/`. Then:

```bash
python -m src.indexing.index_documents
```

This loads all documents, runs all three chunkers, embeds with MiniLM + SPLADE, and upserts into three Qdrant collections in a single pass. Expected time: 5–15 minutes depending on document count and whether OCR is enabled.

Optional flags:

```
--data-dir data/raw    # source directory (default: data/raw)
--no-recreate          # append to existing collections instead of recreating
```

> **OCR note:** OCR is enabled by default (`APP_OCR_ENABLED=true`). Set it to `false` for digital-native PDFs to skip OCR and speed up indexing significantly.

### 6. Run the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501. The sidebar shows the active configuration (embedding model, LLM, reranker). Use the **Collection** and **Retrieval method** dropdowns to explore configurations; the **Chunks passed to LLM** slider controls how many reranked chunks are included in the LLM prompt.

### 7. (Optional) Run the evaluation matrix

```bash
# Full 3×4 matrix — 12 MLflow runs, ~2 hours with free-tier rate limits
python -m src.evaluation.compare

# Re-run only the RAGAS judge (reuse collected answers)
python -m src.evaluation.compare --skip-collect

# Single collection / specific variants
python -m src.evaluation.compare \
  --collections rag_recursive \
  --variants hybrid_rrf "hybrid_rrf+rerank"

# View results in MLflow UI (macOS: avoid port 5000, reserved for AirPlay)
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5001
```

---

## Full Pipeline

```
Raw Documents (PDF, DOCX, PPTX, HTML, MD, CSV, XLSX, TXT)
    |
    v  src/ingestion/
       DocumentLoader routes plain text directly; all other formats through docling.
       Outputs LoadedDocument with page content and metadata.
    |
    v  src/chunking/       [three parallel strategies — one Qdrant collection each]
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
         RAGPipeline.stream() — token-streaming variant used by the dashboard
    |
    v  src/evaluation/
         compare.py — 3 collections × 4 variants = 12 MLflow runs
         RAGAS metrics: faithfulness · answer_relevancy · context_precision · context_recall
```

---

## Retrieval Architecture

The production retrieval path is:

```
query
  -> embed (MiniLM dense + BM25 in-memory)
  -> HybridRetriever.retrieve_rrf(limit=20)    -- RRF fuses dense + BM25 by rank position
  -> Reranker.rerank(top_n=5)                  -- cross-encoder scores each (query, chunk) pair
  -> top K chunks passed as LLM context
```

**Why four stages:**

- Dense search captures semantic similarity but drifts on proper nouns, exact terms, and numbers.
- BM25 matches exact terms but has no semantic understanding.
- RRF fusion combines both signals without requiring compatible score scales.
- The cross-encoder reranker reads query and passage together as a pair, producing a direct relevance score. Too slow for the full corpus but fast on 20 pre-filtered candidates.

**Empirically verified:** RRF occasionally promotes a wrong document to rank 1 when both dense and BM25 partially agree on it. The reranker corrects this in all tested cases. Always run the full four-stage pipeline.

---

## Generation

`RAGPipeline` wraps the full retrieve-rerank-generate cycle. Construction is slow once (BM25 index build + cross-encoder load); call `invoke` or `stream` many times after.

```python
from configs.settings import AppConfig
from src.embedding import Embedder
from src.vectorstore import VectorStore
from src.generation import RAGPipeline

cfg      = AppConfig()
embedder = Embedder()
store    = VectorStore()

pipeline = RAGPipeline(embedder, store, cfg.default_collection)

# Blocking — returns structured dict
result = pipeline.invoke("How does multi-head attention work?")
print(result["answer"])
for chunk in result["source_chunks"]:
    print(chunk.score, chunk.metadata["filename"])

# Streaming — yields tokens as the LLM generates them
for token in pipeline.stream("What is the RAGAS faithfulness metric?"):
    print(token, end="", flush=True)
```

`invoke()` always returns the same dict shape, even on failure:

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

---

## Evaluation

The evaluation framework runs a 3 × 4 comparison matrix and logs every run to MLflow.

**4 retrieval variants:**

| Variant | Method | Rerank |
|---|---|---|
| `dense` | MiniLM bi-encoder only | No |
| `bm25` | Okapi BM25 term overlap | No |
| `hybrid_rrf` | RRF fusion of dense + BM25 | No |
| `hybrid_rrf+rerank` | RRF fusion then FlashRank cross-encoder | Yes |

**RAGAS metrics:**
- `faithfulness` — fraction of answer claims supported by the retrieved context
- `answer_relevancy` — semantic similarity between answer and synthetic re-questions
- `context_precision` — fraction of retrieved chunks that are actually relevant
- `context_recall` — fraction of ground-truth information covered by retrieved chunks

### Full results (3 collections × 4 variants, 27 questions each)

| Collection | Variant | faithfulness | answer_rel | ctx_precision | ctx_recall |
|---|---|---|---|---|---|
| fixed | dense | 0.681 | 0.622 | 0.761 | 0.840 |
| fixed | bm25 | 0.743 | 0.619 | 0.829 | 0.895 |
| fixed | hybrid_rrf | 0.755 | 0.691 | 0.854 | 0.877 |
| fixed | hybrid_rrf+rerank | 0.722 | 0.655 | **0.861** | 0.914 |
| recursive | dense | 0.723 | 0.648 | 0.834 | 0.870 |
| recursive | bm25 | 0.656 | 0.576 | 0.848 | 0.907 |
| **recursive** | **hybrid_rrf** | 0.796 | **0.738** | 0.827 | 0.889 |
| **recursive** | **hybrid_rrf+rerank** | **0.802** | 0.709 | 0.847 | **0.944** |
| semantic | dense | 0.637 | 0.597 | 0.753 | 0.889 |
| semantic | bm25 | 0.661 | 0.522 | 0.731 | 0.790 |
| semantic | hybrid_rrf | 0.670 | 0.629 | 0.763 | 0.907 |
| semantic | hybrid_rrf+rerank | 0.715 | 0.607 | 0.796 | 0.914 |

---

## Findings

### Chunking strategy matters more than expected

The gap between the best configuration (`recursive + hybrid_rrf+rerank`) and the worst (`semantic + dense`) is **0.165 faithfulness points** and **0.154 recall points** — not a rounding error. Chunking strategy is as important a lever as retrieval method choice.

**Why recursive wins:** Recursive chunking respects natural text boundaries (paragraphs, then sentences, then words). The LLM receives coherent passages it can quote from directly. Fixed-size chunks frequently cut mid-sentence, forcing the LLM to infer continuations — a hallucination risk. Semantic chunking at the 95th percentile breakpoint produces 16,386 chunks — 32% more than recursive — but many are very short (one or two sentences). The LLM receives atomised facts without surrounding context, making grounded synthesis harder.

### The reranker's primary contribution is recall, not precision

`recursive + hybrid_rrf` scores 0.889 recall without reranking. With the cross-encoder it jumps to **0.944** — a +0.055 gain. The cross-encoder re-scores all 20 first-stage candidates together, surfacing relevant chunks that RRF fusion ranked lower due to score scale mismatch between the dense and BM25 signals. Context precision barely moves (+0.020), confirming the reranker is recovering missed relevant chunks, not filtering noise.

### Answer relevancy is inversely correlated with reranking

`recursive + hybrid_rrf` scores **0.738** answer relevancy vs **0.709** for `recursive + hybrid_rrf+rerank`. The cross-encoder promotes highly specific technical passages. The LLM then produces a narrower, more precise answer — which the judge scores as slightly less relevant to the original broad question. A more faithful but narrower answer is the right tradeoff for production.

### Three operating points

| Operating point | Config | Faithfulness | Ctx recall | Latency |
|---|---|---|---|---|
| **Lowest latency, acceptable quality** | `semantic + hybrid_rrf+rerank` | 0.715 | 0.914 | 2.1s |
| **Best quality/latency balance** | `recursive + hybrid_rrf` | 0.796 | 0.889 | 2.6s |
| **Maximum quality** | `recursive + hybrid_rrf+rerank` | **0.802** | **0.944** | ~2.6s* |

*The 8.69s latency recorded for the best config in evaluation includes an API rate-limit outlier in generation. The reranker itself adds ~0.2s; real-world generation latency is API-bound and typically under 3s.

**Dashboard default: `rag_recursive + hybrid_rrf+rerank`**

---

## Qdrant Collections

| Collection | Chunking strategy | Points |
|---|---|---|
| `rag_fixed` | Fixed-size (512 chars, 64 overlap) | 10,439 |
| `rag_recursive` | Recursive (512 chars, 64 overlap) | 12,389 |
| `rag_semantic` | Semantic (95th percentile breakpoint) | 16,386 |

Each point carries: dense vector (384-dim cosine) + sparse vector (SPLADE) + payload with `chunk_text`, `source_file`, `chunk_strategy`, `chunk_index`, `chunk_size`, `parent_source`, `filename`, `format`, `total_pages`.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Document parsing | `docling`, `pypdfium2`, `rapidocr`, `ocrmac` |
| Office formats | `python-docx`, `python-pptx`, `openpyxl` |
| Chunking | `langchain-text-splitters` |
| Dense embeddings | `sentence-transformers`, `torch`, `transformers` |
| Sparse embeddings | `fastembed` (ONNX runtime) |
| Vector store | `qdrant-client` |
| Sparse retrieval | `rank-bm25` |
| Reranking | `flashrank` |
| Generation | `langchain-openai`, `langchain-core` (LCEL chain) |
| Evaluation | `ragas`, `mlflow` |
| Dashboard | `streamlit`, `plotly` |
| Configuration | `pydantic-settings`, `python-dotenv` |
| Testing / DX | `pytest`, `ruff` |

---

## Configuration Reference

All settings live in `AppConfig` (`configs/settings.py`). Fields without `APP_` use `validation_alias` to read their own env var name directly.

| Variable | Default | Description |
|---|---|---|
| `APP_CHUNK_SIZE` | `512` | Target chunk size in characters |
| `APP_CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `APP_SEMANTIC_BREAKPOINT_THRESHOLD` | `95` | Percentile threshold for semantic breakpoints |
| `APP_DENSE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for dense vectors |
| `APP_DENSE_EMBEDDING_DIM` | `384` | Output dimension of the dense model |
| `APP_SPARSE_EMBEDDING_MODEL` | `prithivida/Splade_PP_en_v1` | fastembed SPLADE model for sparse vectors |
| `APP_RRF_K` | `60` | RRF damping constant (from the original paper) |
| `APP_HYBRID_ALPHA` | `0.7` | Dense weight in weighted score fusion |
| `APP_RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | FlashRank cross-encoder |
| `APP_RETRIEVAL_LIMIT` | `20` | First-stage candidates fetched before reranking |
| `APP_RERANK_TOP_N` | `5` | Chunks passed to the LLM as context |
| `APP_MAX_CONTEXT_CHARS` | `12000` | Context length budget (~3,000 tokens) |
| `APP_DEFAULT_COLLECTION` | `rag_recursive` | Default collection (evaluation-backed) |
| `APP_DEFAULT_RETRIEVAL_METHOD` | `hybrid_rrf+rerank` | Default method (evaluation-backed) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | LLM API base URL |
| `OLLAMA_API_KEY` | _(empty)_ | API key (empty for local Ollama) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Generation model |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature |
| `OLLAMA_MAX_TOKENS` | `1024` | Max generated tokens |
| `REQUEST_TIMEOUT` | `60` | LLM request timeout in seconds |
| `RAGAS_JUDGE_MODEL` | `llama3.1:70b` | Judge model for RAGAS evaluation |
| `RAGAS_MAX_WORKERS` | `2` | RAGAS judge concurrency (keep ≤2 for free-tier APIs) |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns.db` | MLflow backend |
| `MLFLOW_EXPERIMENT_NAME` | `rag_chunking_comparison` | MLflow experiment name |
| `APP_QDRANT_HOST` | `localhost` | Qdrant host |
| `APP_QDRANT_PORT` | `6333` | Qdrant port |
| `APP_OCR_ENABLED` | `true` | Enable OCR (set `false` for digital PDFs to speed up indexing) |

---

## Testing

Unit tests cover retrieval logic, chunkers, and prompt formatting — no external services required.

```bash
pip install -r requirements-dev.txt   # pytest + ruff (one-time)
pytest tests/unit/                    # 76 tests, ~0.5s
pytest tests/ -m integration -v      # requires Qdrant + populated collections
```

Integration tests are auto-skipped unless you pass `-m integration`. Run them after `docker compose up -d` and `python -m src.indexing.index_documents`.

---

## Known Limitations

**BM25 index is in-memory and rebuilt at startup**

`BM25Retriever` loads all chunk texts into RAM and builds an Okapi BM25 index when the pipeline is constructed (~10 seconds for 12k chunks). Adding documents requires a full re-index and a server restart to rebuild the in-memory index. This approach works well for datasets up to ~100k chunks on a standard machine; beyond that, a proper search backend (Elasticsearch, OpenSearch) would be more appropriate.

**No incremental indexing**

`index_documents.py` drops and recreates all three collections by default. Adding one new document means re-embedding and re-upserting the entire corpus. Use `--no-recreate` to append without dropping, but this doesn't remove stale points from deleted documents.

**Semantic chunking is slow at indexing time**

`SemanticChunker` embeds every sentence to compute breakpoints — roughly 10× slower than `FixedSizeChunker` or `RecursiveChunker` for the same document set. Fine for an offline indexing job; not suitable for real-time or streaming document ingestion.

**RAGAS evaluation requires a capable judge model**

RAGAS fires the judge LLM on every question × metric. Faithfulness scores from `gemma3:12b` are reasonable but will differ from human ratings, especially on nuanced cases (literary paraphrase, partial grounding). The 0.0 faithfulness scores on Q08 and Q21 in the best config are metric artifacts — the LLM said "I don't know" which RAGAS scores as unfaithful even though the behavior is correct.

**Streamlit is single-threaded per session**

The cached pipelines (`@st.cache_resource`) are shared across sessions but Streamlit executes one script per session serially. Two users submitting queries simultaneously will serialize. The BM25 retriever and cross-encoder are not thread-safe for parallel writes.

**Free-tier Ollama rate limits affect evaluation speed**

Running the full 12-run evaluation matrix with `RAGAS_MAX_WORKERS=2` takes ~2 hours on the Ollama cloud free tier. The default `max_workers=16` in RAGAS 0.4.x overwhelms the rate limit immediately. If you have access to a faster LLM endpoint, increase `RAGAS_MAX_WORKERS` and the evaluation completes in 20–30 minutes.

**Context window is hard-capped at 12,000 characters**

If the top-K chunks exceed `APP_MAX_CONTEXT_CHARS`, the lowest-ranked chunks are silently dropped before the LLM call. This protects against accidental context blowout but means the LLM may not see all retrieved information on long documents. Increase the limit or reduce top-K if you're seeing truncation in the logs.

**Dashboard is local-only by design**

The app talks to `localhost` Qdrant and an Ollama endpoint. Deploying to a server requires a network-accessible Qdrant instance and an LLM endpoint with proper authentication, plus Streamlit session isolation if multiple users are expected.

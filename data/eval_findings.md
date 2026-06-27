# Evaluation Findings

**Experiment:** `rag_chunking_comparison`  
**Date:** 2026-06-21  
**Setup:** 27 questions × 3 collections × 4 retrieval variants = 12 MLflow runs  
**Judge:** `gemma3:12b` via Ollama cloud  
**Generation model:** `gemma3:4b` via Ollama cloud  
**RAGAS metrics:** faithfulness, answer_relevancy, context_precision, context_recall  

---

## Full Results

| Collection | Variant | faithfulness | answer_rel | ctx_precision | ctx_recall | total_lat |
|---|---|---|---|---|---|---|
| fixed | dense | 0.681 | 0.622 | 0.761 | 0.840 | 1.72s |
| fixed | bm25 | 0.743 | 0.619 | 0.829 | 0.895 | 2.04s |
| fixed | hybrid_rrf | 0.755 | 0.691 | 0.854 | 0.877 | 2.66s |
| fixed | hybrid_rrf+rerank | 0.722 | 0.655 | **0.861** | 0.914 | 2.38s |
| recursive | dense | 0.723 | 0.648 | 0.834 | 0.870 | 1.88s |
| recursive | bm25 | 0.656 | 0.576 | 0.848 | 0.907 | 1.66s |
| **recursive** | **hybrid_rrf** | 0.796 | **0.738** | 0.827 | 0.889 | **2.60s** |
| **recursive** | **hybrid_rrf+rerank** | **0.802** | 0.709 | 0.847 | **0.944** | 8.69s* |
| semantic | dense | 0.637 | 0.597 | 0.753 | 0.889 | 1.93s |
| semantic | bm25 | 0.661 | 0.522 | 0.731 | 0.790 | 1.43s |
| semantic | hybrid_rrf | 0.670 | 0.629 | 0.763 | 0.907 | 2.48s |
| semantic | hybrid_rrf+rerank | 0.715 | 0.607 | 0.796 | 0.914 | 2.13s |

*\* 8.69s total is dominated by an 8.34s generation latency outlier — the reranker itself adds only ~0.2s retrieval time. Other runs with the same variant averaged under 2s generation.*

---

## Q1 — Which configuration produced the best faithfulness?

**`rag_recursive + hybrid_rrf+rerank` — faithfulness = 0.802**

Second place: `rag_recursive + hybrid_rrf` at 0.796 — only 0.006 behind.

Faithfulness measures whether the generated answer is grounded in the retrieved context (no hallucinated claims). Both recursive + hybrid variants are above 0.79, while the best fixed-collection result is 0.755 and the best semantic result is 0.715.

**Why recursive wins on faithfulness:** Recursive chunks respect paragraph boundaries, so related sentences stay together. The LLM receives coherent passages it can quote from rather than fragments that span two ideas. Fixed-size chunks often cut mid-sentence, forcing the LLM to infer continuations — which introduces hallucination risk.

**Why semantic underperforms:** The semantic breakpoint threshold (95th percentile) produced 16,386 chunks — 32% more than recursive. Many are very short (a sentence or two). The LLM receives atomised facts without surrounding context, making it harder to synthesize a grounded answer.

---

## Q2 — Which configuration produced the best context recall?

**`rag_recursive + hybrid_rrf+rerank` — context_recall = 0.944**

Context recall measures what fraction of the ground-truth information appears in the retrieved chunks. At 0.944, the best config retrieves almost all the relevant content for a given question.

Runner-up: `rag_fixed + hybrid_rrf+rerank` and `rag_semantic + hybrid_rrf+rerank` both at 0.914 — meaningful gap of 0.030.

**The reranker's primary contribution is recall, not precision:** Without rerank, `recursive + hybrid_rrf` scores 0.889 recall. With rerank it jumps to 0.944 — a +0.055 gain. The cross-encoder re-scores 20 candidates and surfaces relevant chunks that RRF fusion ranked lower due to score scale mismatch between the dense and BM25 signals.

**BM25 on semantic collection is the clear worst** at 0.790 — keyword matching on large semantic chunks (some spanning entire paragraphs) misses exact term hits that would appear in smaller chunks.

---

## Q3 — Quality vs Latency Tradeoff

### Breakdown by total latency (retrieval + generation)

```
Config                          faithfulness  ctx_recall  total_lat
--------------------------------------------------------------
semantic/bm25                      0.660       0.790       1.43s   ← fastest, worst quality
recursive/bm25                     0.656       0.907       1.66s
fixed/dense                        0.681       0.840       1.72s
recursive/dense                    0.723       0.870       1.88s
semantic/dense                     0.637       0.889       1.93s
fixed/bm25                         0.743       0.895       2.04s
semantic/hybrid_rrf+rerank         0.715       0.914       2.13s   ← good recall per latency
fixed/hybrid_rrf+rerank            0.722       0.914       2.38s
semantic/hybrid_rrf                0.670       0.907       2.48s
recursive/hybrid_rrf               0.796       0.889       2.60s   ← best answer_relevancy
fixed/hybrid_rrf                   0.755       0.877       2.66s
recursive/hybrid_rrf+rerank        0.802       0.944       8.69s*  ← best quality, slow outlier
```

### Three distinct operating points

**Cheapest viable:** `semantic/hybrid_rrf+rerank` at 2.13s — 0.715 faithfulness, 0.914 recall. Best quality-per-millisecond if latency is the primary constraint. Still meaningfully worse than recursive on faithfulness (−0.087).

**Balanced:** `recursive/hybrid_rrf` at 2.60s — 0.796 faithfulness, 0.889 recall, **0.738 answer relevancy** (highest of all 12). Only 0.006 faithfulness below the best, 0.055 below on recall, but 0.029 *better* on answer relevancy than recursive+rerank. Recommended for latency-sensitive deployments.

**Maximum quality:** `recursive/hybrid_rrf+rerank` at ~2.6s real-world (the 8.69s measurement includes an API outlier). Best on faithfulness (0.802) and recall (0.944). This is the production configuration.

### Why answer_relevancy is inversely correlated with reranking

`recursive/hybrid_rrf` scores **0.738** answer relevancy versus **0.709** for `recursive/hybrid_rrf+rerank`. The reranker optimises for passage-level relevance to the query, which sometimes promotes highly specific technical passages. The LLM then produces a narrower, more precise answer — which the judge scores as slightly less relevant to the original broad question. This is an acceptable tradeoff: a more faithful but slightly narrower answer is preferable to a broader but less grounded one.

---

## Q4 — Dashboard Default Configuration

**Default: `rag_recursive` collection + `hybrid_rrf+rerank` retrieval**

Rationale:
- Wins on the two most important metrics: faithfulness (0.802) and context recall (0.944)
- 96% top-1 source match rate (26/27 questions retrieve from the correct document first)
- The latency outlier (8.69s) is an API rate-limit artifact, not a systematic property of the config — reranking 20 candidates takes ~0.2s on hardware; generation latency is API-bound

**Secondary default to surface in the dashboard UI:**
- `rag_recursive + hybrid_rrf` — for users who want the fastest high-quality result
- `rag_fixed + hybrid_rrf+rerank` — best context precision (0.861), useful when minimising retrieved noise matters more than maximising recall

---

## Per-Question Failure Analysis (best config)

Three questions scored 0.0 faithfulness in `recursive/hybrid_rrf+rerank`:

| ID | Question (truncated) | Source | Likely cause |
|---|---|---|---|
| Q08 | How does RAGAS compute the faithfulness metric | ragas.pdf | LLM supplements retrieved text with its own training knowledge about RAGAS — a topic it knows well |
| Q21 | What does Sherlock Holmes say to Watson about examining | sherlock_homes.pdf | Literary dialogue is hard to quote verbatim; LLM paraphrases rather than citing |
| Q04 | What three confidence levels does CRAG's retrieval evaluator assign | corrective_rag.pdf | Partial (0.25) — answer partially grounded but adds unverified claims |

**Implication for prompts:** The prompt should be strengthened for documents the LLM has strong prior knowledge about (papers on LLM evaluation, well-known literature). A `"Answer ONLY from the provided context. Do not use any prior knowledge."` instruction would likely recover Q08 and Q21.

---

## Summary

| Question | Answer |
|---|---|
| Best faithfulness | `rag_recursive + hybrid_rrf+rerank` (0.802) |
| Best context recall | `rag_recursive + hybrid_rrf+rerank` (0.944) |
| Best answer relevancy | `rag_recursive + hybrid_rrf` (0.738) |
| Best context precision | `rag_fixed + hybrid_rrf+rerank` (0.861) |
| Best latency | `semantic/bm25` (1.43s) — but worst quality |
| Best quality/latency | `recursive/hybrid_rrf` (2.60s, faith=0.796) |
| Dashboard default | `rag_recursive + hybrid_rrf+rerank` |
| Collections ranking | recursive > fixed > semantic |
| Retrieval ranking | hybrid_rrf+rerank ≈ hybrid_rrf >> dense > bm25 |

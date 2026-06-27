from __future__ import annotations

import logging
from typing import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from configs.settings import AppConfig
from src.embedding.embedder import Embedder
from src.generation.prompts import CONCISE_PROMPT, DETAILED_PROMPT, format_context
from src.models import RetrievalResult
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import BM25Retriever
from src.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

_RETRIEVAL_METHODS = frozenset({"hybrid_rrf", "hybrid_weighted", "dense", "bm25"})

# Llama tokenizer averages 3.5–4 chars/token; 4 is a conservative lower bound
# used for the context-length budget check.
_CHARS_PER_TOKEN = 4


class RAGPipeline:
    """Full retrieve-rerank-generate pipeline backed by Ollama.

    Construction is slow once: the BM25 index build (~10 s for 12 k chunks)
    and the FlashRank cross-encoder load happen here. Construct once, call
    ``invoke`` or ``stream`` many times.

    ``invoke`` always returns the same dict shape — even on LLM failure or
    empty retrieval — so evaluation loops can iterate without guarding every
    call::

        result = pipeline.invoke(question)
        if result["error"]:
            log_failure(result)
        else:
            score(result["answer"], result["source_chunks"])

    ``stream`` yields tokens as the LLM generates them. Retrieval is blocking
    (must complete before the prompt can be built). LLM timeout errors during
    streaming propagate to the caller as exceptions — the generator protocol
    has no clean way to embed a structured error mid-stream.

    Usage::

        pipeline = RAGPipeline(embedder, store, cfg.collection_recursive)

        result = pipeline.invoke("What is the RAGAS faithfulness metric?")
        print(result["answer"])
        for chunk in result["source_chunks"]:
            print(chunk.metadata["filename"], chunk.score)

        for token in pipeline.stream("How does multi-head attention work?"):
            print(token, end="", flush=True)
        print()
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        collection: str,
        config: AppConfig | None = None,
        use_detailed_prompt: bool = True,
    ) -> None:
        cfg = config or AppConfig()
        self._cfg = cfg
        self._collection = collection

        logger.info(
            "Building RAGPipeline  collection=%s  model=%s  url=%s",
            collection,
            cfg.ollama_model,
            cfg.ollama_base_url,
        )

        self._dense    = DenseRetriever(embedder, store)
        self._bm25     = BM25Retriever(store, collection)
        self._hybrid   = HybridRetriever(self._dense, self._bm25, cfg)
        self._reranker = Reranker(cfg)

        # Ollama cloud exposes an OpenAI-compatible endpoint at /v1/.
        # ChatOpenAI sends Authorization: Bearer <key> which Ollama cloud requires.
        # Local Ollama also supports /v1/ so this works for both deployments.
        llm = ChatOpenAI(
            base_url=cfg.ollama_base_url.rstrip("/") + "/v1/",
            api_key=cfg.ollama_api_key or "ollama",
            model=cfg.ollama_model,
            temperature=cfg.ollama_temperature,
            max_tokens=cfg.ollama_max_tokens,
            timeout=cfg.request_timeout,
        )

        # Dashboard uses DETAILED_PROMPT for stronger grounding instructions.
        # Evaluation passes use_detailed_prompt=False to reduce token usage.
        prompt = DETAILED_PROMPT if use_detailed_prompt else CONCISE_PROMPT
        self._llm_chain = prompt | llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(
        self,
        question: str,
        collection: str | None = None,
        retrieval_method: str = "hybrid_rrf",
    ) -> dict:
        """Run the full pipeline and return a structured result dict.

        Always returns::

            {
                "answer":           str,
                "source_chunks":    list[RetrievalResult],
                "retrieval_method": str,
                "collection":       str,
                "query":            str,
                "error":            str | None,   # None on success
            }

        ``error`` is ``None`` on a successful run. On failure it contains the
        exception message; ``answer`` contains a human-readable failure notice.
        ``source_chunks`` is always populated with whatever was retrieved
        (empty list if retrieval returned nothing) so callers can inspect
        what the pipeline saw regardless of whether generation succeeded.

        Args:
            question:         The user's question.
            collection:       Override the default collection set at construction.
            retrieval_method: One of "hybrid_rrf", "hybrid_weighted", "dense", "bm25".
        """
        col = collection or self._collection

        # --- stage 1: retrieval ---
        chunks = self._retrieve(question, col, retrieval_method)

        if not chunks:
            logger.warning(
                "Empty retrieval  question=%r  collection=%s  method=%s",
                question[:80], col, retrieval_method,
            )
            return {
                "answer": (
                    "No relevant context was found in the documents for this query. "
                    "Try rephrasing or broadening the question."
                ),
                "source_chunks": [],
                "retrieval_method": retrieval_method,
                "collection": col,
                "query": question,
                "error": None,
            }

        # --- stage 2: context length guard ---
        chunks = self._truncate_to_limit(chunks, question)
        context = format_context(chunks)

        # --- stage 3: generation ---
        try:
            answer = self._llm_chain.invoke({"context": context, "question": question})
            error: str | None = None
        except Exception as exc:
            logger.error(
                "LLM call failed  question=%r  error=%s", question[:80], exc,
            )
            answer = "Generation failed: the language model did not return a response."
            error = str(exc)

        logger.info(
            "invoke complete  method=%s  chunks=%d  error=%s",
            retrieval_method, len(chunks), error,
        )
        return {
            "answer": answer,
            "source_chunks": chunks,
            "retrieval_method": retrieval_method,
            "collection": col,
            "query": question,
            "error": error,
        }

    def stream(
        self,
        question: str,
        collection: str | None = None,
        retrieval_method: str = "hybrid_rrf",
    ) -> Iterator[str]:
        """Retrieve context (blocking), then stream LLM tokens as they arrive.

        Empty retrieval yields a single informational string and returns.
        LLM timeout or API errors during streaming propagate as exceptions —
        wrap the call site in try/except if the dashboard needs to handle them.
        """
        col = collection or self._collection
        chunks = self._retrieve(question, col, retrieval_method)

        if not chunks:
            logger.warning(
                "Empty retrieval  question=%r  collection=%s  method=%s",
                question[:80], col, retrieval_method,
            )
            yield (
                "No relevant context was found in the documents for this query. "
                "Try rephrasing or broadening the question."
            )
            return

        chunks = self._truncate_to_limit(chunks, question)
        context = format_context(chunks)
        logger.info(
            "stream started  method=%s  chunks=%d", retrieval_method, len(chunks),
        )
        try:
            yield from self._llm_chain.stream({"context": context, "question": question})
        except Exception as exc:
            logger.error("LLM stream failed  question=%r  error=%s", question[:80], exc)
            yield f"\n\n[Generation failed: {exc}]"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        question: str,
        collection: str,
        retrieval_method: str,
        rerank: bool = True,
    ) -> list[RetrievalResult]:
        """Dispatch to the requested retrieval strategy, optionally rerank.

        ``rerank=False`` is used by the evaluation comparison matrix to measure
        retrieval quality before the cross-encoder step, isolating its contribution.
        Production calls always use ``rerank=True``.
        """
        if retrieval_method not in _RETRIEVAL_METHODS:
            raise ValueError(
                f"Unknown retrieval_method {retrieval_method!r}. "
                f"Choose from: {sorted(_RETRIEVAL_METHODS)}"
            )

        limit = self._cfg.retrieval_limit

        if retrieval_method == "hybrid_rrf":
            candidates = self._hybrid.retrieve_rrf(question, collection, limit=limit)
        elif retrieval_method == "hybrid_weighted":
            candidates = self._hybrid.retrieve_weighted(question, collection, limit=limit)
        elif retrieval_method == "dense":
            candidates = self._dense.retrieve(question, collection, limit=limit)
        else:  # bm25
            candidates = self._bm25.retrieve(question, limit=limit)

        if rerank:
            return self._reranker.rerank(question, candidates, top_n=self._cfg.rerank_top_n)
        return candidates[: self._cfg.rerank_top_n]

    def _truncate_to_limit(
        self,
        chunks: list[RetrievalResult],
        question: str,
    ) -> list[RetrievalResult]:
        """Drop lowest-ranked chunks until total context fits within budget.

        Uses a character-count proxy for token count (1 token ~ 4 chars).
        Only the chunk body text is counted; prompt overhead is ~150 tokens
        and is well within the headroom this limit provides.
        """
        limit = self._cfg.max_context_chars
        total_chars = sum(len(r.chunk_text) for r in chunks)

        if total_chars <= limit:
            return chunks

        kept = list(chunks)
        while kept and sum(len(r.chunk_text) for r in kept) > limit:
            kept.pop()

        dropped = len(chunks) - len(kept)
        logger.warning(
            "Context truncated: dropped %d chunk(s)  "
            "original_chars=%d (~%d tokens)  limit_chars=%d  question=%r",
            dropped,
            total_chars,
            total_chars // _CHARS_PER_TOKEN,
            limit,
            question[:60],
        )
        return kept

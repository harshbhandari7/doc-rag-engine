from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from configs.settings import AppConfig

logger = logging.getLogger(__name__)


def make_judge(
    config: AppConfig | None = None,
) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Build the RAGAS judge LLM and embeddings from project config.

    Uses Ollama's OpenAI-compatible endpoint so the same API key and base URL
    work for both generation (pipeline.py) and evaluation.

    Returns:
        (judge_llm, judge_embeddings) — pass directly to ragas_evaluate() as
        ``llm=`` and ``embeddings=``.
    """
    cfg = config or AppConfig()

    logger.info(
        "Creating RAGAS judge  judge_model=%s  embed_model=%s  url=%s",
        cfg.ragas_judge_model,
        cfg.dense_embedding_model,
        cfg.ollama_base_url,
    )

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=cfg.ollama_base_url.rstrip("/") + "/v1/",
            api_key=cfg.ollama_api_key or "ollama",
            model=cfg.ragas_judge_model,
            temperature=0.0,
        )
    )

    # Same embedding model used for retrieval — keeps AnswerRelevancy scores
    # in the same vector space as the retrieval signal.
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=cfg.dense_embedding_model)
    )

    return judge_llm, judge_embeddings

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.models import RetrievalResult


# ---------------------------------------------------------------------------
# Context formatter
#
# Turns a list of RetrievalResult objects into the {context} string that
# both prompt templates slot in. Kept here because formatting is a prompt
# engineering decision — changing how sources are labelled or how chunks
# are separated affects what the LLM sees, not how the pipeline routes data.
#
# Format chosen: numbered passages so the LLM can cite [1], [2] inline,
# with source attribution on the same header line for traceability.
# ---------------------------------------------------------------------------

def format_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a numbered, attributed context block.

    Example output::

        [1] Source: attention_need.pdf | Chunk 14
        Multi-head attention allows the model to jointly attend to information
        from different representation subspaces at different positions.

        [2] Source: ragas.pdf | Chunk 2
        Faithfulness measures how factually consistent the generated answer
        is with the retrieved context passages.

    Args:
        results: Reranked ``RetrievalResult`` objects, ordered by relevance.

    Returns:
        A single string ready to be inserted into the ``{context}`` slot of
        either prompt template.
    """
    parts = []
    for i, r in enumerate(results, 1):
        filename  = r.metadata.get("filename", "unknown")
        chunk_idx = r.metadata.get("chunk_index", "?")
        header    = f"[{i}] Source: {filename} | Chunk {chunk_idx}"
        parts.append(f"{header}\n{r.chunk_text.strip()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt 1: Concise
#
# Shorter system message — lower token usage, faster responses.
# Best for evaluation runs where the pipeline is called many times and
# throughput matters more than exhaustive grounding instructions.
# ---------------------------------------------------------------------------

CONCISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a question-answering assistant. "
        "Answer using only the numbered context passages provided. "
        "Cite sources inline by number, e.g. [1] or [2]. "
        "If the context does not contain the answer, respond with exactly: "
        "\"The context does not contain enough information to answer this question.\"",
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}",
    ),
])


# ---------------------------------------------------------------------------
# Prompt 2: Detailed
#
# Explicit step-by-step instructions with a structured response format.
# Asks the LLM to reason over passages before committing to an answer,
# which improves faithfulness on complex multi-hop questions.
# Best for the dashboard demo where answer quality matters most.
#
# Response format requested:
#   Reasoning: <which passages apply and why>
#   Answer:    <grounded answer with inline citations>
# ---------------------------------------------------------------------------

DETAILED_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise question-answering assistant. "
        "Answer questions using only the numbered context passages provided below. "
        "Follow these steps:\n\n"
        "1. Read every passage before forming an answer.\n"
        "2. Identify which passages are directly relevant to the question.\n"
        "3. Base your answer solely on those passages. Do not use outside knowledge.\n"
        "4. Cite the source of each claim inline by passage number, "
        "e.g. \"transformers use self-attention [1] which can be stacked [2].\"\n"
        "5. If multiple passages support the same claim, cite all of them, e.g. [1][3].\n"
        "6. If the context does not contain enough information, respond with exactly:\n"
        "   \"The context does not contain enough information to answer this question.\"\n\n"
        "Format your response as:\n"
        "Reasoning: <brief note on which passages are relevant and why>\n"
        "Answer: <your grounded, cited answer>",
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}",
    ),
])

from pydantic import Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Single project-wide configuration.

    Most fields are overridden via an ``APP_``-prefixed env var
    (e.g. ``APP_LOG_LEVEL=DEBUG``, ``APP_OCR_ENABLED=false``).

    Ollama fields use their own un-prefixed names (``OLLAMA_BASE_URL`` etc.)
    because they match the convention of the Ollama cloud API — see ``.env``.
    ``validation_alias`` tells pydantic-settings to look up the exact env var
    name for those fields, bypassing the ``APP_`` prefix.
    """

    # --- general ---
    log_level: str = "INFO"
    data_dir: str = "data/raw"

    # --- ingestion: docling pipeline ---
    ocr_enabled: bool = True
    table_structure_enabled: bool = True
    document_timeout: float | None = None  # seconds; None = no limit

    # --- ingestion: page filtering & merging ---
    min_page_chars: int = 10  # pages with fewer chars are treated as empty
    merge_min_chars: int = 50  # adjacent pages shorter than this get merged

    # --- chunking ---
    chunk_size: int = 512    # target chunk size in characters
    chunk_overlap: int = 64  # overlap in characters between consecutive chunks
    semantic_breakpoint_threshold: int = 75  # percentile; higher = fewer, larger semantic chunks

    # --- embedding ---
    dense_embedding_model: str = "all-MiniLM-L6-v2"          # sentence-transformers, dense vectors
    dense_embedding_dim: int = 384                           # output dim of dense_embedding_model
    sparse_embedding_model: str = "prithivida/Splade_PP_en_v1"  # fastembed, sparse vectors

    # --- hybrid retrieval ---
    rrf_k: int = 60          # RRF damping constant; from the original paper, 60 works well in practice
    hybrid_alpha: float = 0.7  # weighted fusion: alpha * dense + (1-alpha) * sparse; 0.7 = 70% semantic

    # --- reranking ---
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"  # FlashRank cross-encoder; ~90MB, downloads to /tmp on first use

    # --- generation: ollama ---
    # These fields use validation_alias so pydantic-settings reads the exact
    # env var name shown, ignoring the APP_ prefix applied to all other fields.
    ollama_api_key: str = Field(default="", validation_alias="OLLAMA_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", validation_alias="OLLAMA_MODEL")
    ollama_temperature: float = Field(default=0.1, validation_alias="OLLAMA_TEMPERATURE")
    ollama_max_tokens: int = Field(default=1024, validation_alias="OLLAMA_MAX_TOKENS")
    request_timeout: int = Field(default=60, validation_alias="REQUEST_TIMEOUT")
    # MLflow experiment tracking
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlruns.db", validation_alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="rag_chunking_comparison", validation_alias="MLFLOW_EXPERIMENT_NAME"
    )

    # Judge LLM used by RAGAS during evaluation — separate from the generation model
    # so the two can be swapped independently. Larger judge = more reliable scores.
    ragas_judge_model: str = Field(default="llama3.1:70b", validation_alias="RAGAS_JUDGE_MODEL")

    # RAGAS judge concurrency — default 16 overwhelms free-tier rate limits.
    # 2 keeps requests serialised enough to stay under RPM caps.
    ragas_max_workers: int = 2

    # Evaluation-backed defaults: rag_recursive + hybrid_rrf+rerank scored highest
    # on faithfulness (0.802) and context recall (0.944) across all 12 configurations.
    default_collection: str = "rag_recursive"
    default_retrieval_method: str = "hybrid_rrf+rerank"

    retrieval_limit: int = 20                # hybrid candidates fetched before reranking
    rerank_top_n: int = 5                    # chunks passed to the LLM as context
    # Character budget for the context block sent to the LLM.
    # At ~4 chars/token this is ~3 000 tokens — well within Llama 3.1's window
    # but a hard ceiling that prevents accidental context blowout.
    max_context_chars: int = 12000

    # --- vector store: qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # One collection per chunking strategy so retrieval quality can be compared.
    collection_fixed: str = "rag_fixed"
    collection_recursive: str = "rag_recursive"
    collection_semantic: str = "rag_semantic"

    model_config = {
        "env_prefix": "APP_",
        "populate_by_name": True,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

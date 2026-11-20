"""Central configuration loaded from environment variables / .env file."""
import os
from typing import Optional
from pathlib import Path

# Try importing from pydantic_settings; fallback to pydantic.BaseModel if missing
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as _BaseModel
    _HAS_PYDANTIC_SETTINGS = False


def _load_dotenv_dict(env_path: str = ".env") -> dict:
    """Parse a simple .env file into a dictionary without requiring python-dotenv."""
    data = {}
    path = Path(env_path)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            data[k] = v
    return data


if _HAS_PYDANTIC_SETTINGS:
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")

        # Qdrant
        qdrant_url: str = "http://localhost:6333"
        qdrant_collection: str = "tech_docs"
        qdrant_vector_size: int = 1024

        # Redis
        redis_url: str = "redis://localhost:6379"

        # Models
        embed_model: str = "BAAI/bge-large-en-v1.5"
        rerank_model: str = "BAAI/bge-reranker-large"
        device: str = "auto"  # 'auto', 'cuda', 'mps', 'cpu'
        embed_batch_size: int = 32
        query_instruction: str = "Represent this sentence for searching relevant passages: "

        # LLM
        hf_token: str = ""
        llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
        openai_api_key: str = ""

        # Retrieval
        dense_top_k: int = 20
        sparse_top_k: int = 20
        rerank_top_n: int = 5
        dense_weight: float = 1.0
        sparse_weight: float = 1.0
        rrf_k: int = 60

        # Semantic Cache
        cache_similarity_threshold: float = 0.92
        cache_ttl_seconds: int = 86400  # 24 hours

        # Data directories
        data_raw_dir: str = "data/raw"
        data_processed_dir: str = "data/processed"
        chunks_file: str = "data/processed/chunks.jsonl"
        bm25_index_file: str = "data/processed/bm25_index.pkl"

else:
    class Settings(_BaseModel):
        # Qdrant
        qdrant_url: str = "http://localhost:6333"
        qdrant_collection: str = "tech_docs"
        qdrant_vector_size: int = 1024

        # Redis
        redis_url: str = "redis://localhost:6379"

        # Models
        embed_model: str = "BAAI/bge-large-en-v1.5"
        rerank_model: str = "BAAI/bge-reranker-large"
        device: str = "auto"
        embed_batch_size: int = 32
        query_instruction: str = "Represent this sentence for searching relevant passages: "

        # LLM
        hf_token: str = ""
        llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
        openai_api_key: str = ""

        # Retrieval
        dense_top_k: int = 20
        sparse_top_k: int = 20
        rerank_top_n: int = 5
        dense_weight: float = 1.0
        sparse_weight: float = 1.0
        rrf_k: int = 60

        # Semantic Cache
        cache_similarity_threshold: float = 0.92
        cache_ttl_seconds: int = 86400

        # Data directories
        data_raw_dir: str = "data/raw"
        data_processed_dir: str = "data/processed"
        chunks_file: str = "data/processed/chunks.jsonl"
        bm25_index_file: str = "data/processed/bm25_index.pkl"

        def __init__(self, **kwargs):
            env_vars = _load_dotenv_dict(".env")
            # Override with os.environ
            env_vars.update(os.environ)

            mapped_kwargs = {}
            for field_name, field_info in self.__class__.model_fields.items():
                env_key = field_name.upper()
                val = kwargs.get(field_name) or env_vars.get(env_key) or env_vars.get(field_name)
                if val is not None:
                    field_type = field_info.annotation
                    try:
                        if field_type is int:
                            val = int(val)
                        elif field_type is float:
                            val = float(val)
                        elif field_type is bool:
                            val = str(val).lower() in ("true", "1", "yes")
                    except (ValueError, TypeError):
                        pass
                    mapped_kwargs[field_name] = val

            super().__init__(**mapped_kwargs)


settings = Settings()

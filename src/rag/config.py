"""Central configuration loaded from environment variables / .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "tech_docs"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Models
    embed_model: str = "BAAI/bge-large-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-large"

    # LLM
    hf_token: str = ""
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Retrieval
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_n: int = 5
    cache_similarity_threshold: float = 0.92


settings = Settings()

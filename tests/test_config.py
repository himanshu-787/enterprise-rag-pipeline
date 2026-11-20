"""Unit tests for configuration loading and validation."""
import os
import pytest
from src.rag.config import Settings, settings


def test_default_config():
    """Verify default configuration values."""
    assert settings.qdrant_collection == "tech_docs"
    assert settings.qdrant_vector_size == 1024
    assert "bge-large" in settings.embed_model
    assert "bge-reranker" in settings.rerank_model
    assert settings.dense_top_k == 20
    assert settings.sparse_top_k == 20
    assert settings.rerank_top_n == 5
    assert settings.cache_similarity_threshold == 0.92


def test_env_override(monkeypatch):
    """Verify environment variables override defaults."""
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_docs")
    monkeypatch.setenv("DENSE_TOP_K", "35")
    monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.95")

    custom_settings = Settings()
    assert custom_settings.qdrant_collection == "custom_docs"
    assert custom_settings.dense_top_k == 35
    assert custom_settings.cache_similarity_threshold == 0.95


def test_config_paths():
    """Verify data paths are configured."""
    assert settings.data_raw_dir.replace("\\", "/").endswith("data/raw")
    assert settings.data_processed_dir.replace("\\", "/").endswith("data/processed")

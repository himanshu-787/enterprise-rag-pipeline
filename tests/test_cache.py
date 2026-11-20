"""Unit tests for semantic caching and similarity lookup."""
import time
from src.rag.cache.semantic_cache import SemanticCache


def test_cache_exact_hit():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.clear()

    q = "how to handle errors in fastapi"
    response = {"answer": "Use HTTPException", "sources": ["err_1"]}
    cache.store(q, response)

    hit_val, sim = cache.lookup(q)
    assert hit_val is not None
    assert hit_val["answer"] == "Use HTTPException"
    assert sim >= 0.99


def test_cache_semantic_similarity_hit():
    cache = SemanticCache(similarity_threshold=0.85)
    cache.clear()

    q1 = "how to validate query parameters"
    response = {"answer": "Use Query(..., min_length=3)", "sources": ["c_qp"]}
    cache.store(q1, response)

    q2 = "how do I validate query params"
    hit_val, sim = cache.lookup(q2)
    assert hit_val is not None
    assert sim >= 0.85
    assert hit_val["answer"] == response["answer"]


def test_cache_divergence_miss():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.clear()

    q1 = "how to configure database session"
    cache.store(q1, {"answer": "Use SessionLocal"})

    q2 = "how to run background tasks"
    hit_val, sim = cache.lookup(q2)
    assert hit_val is None
    assert sim < 0.90


def test_cache_ttl_expiration():
    cache = SemanticCache(similarity_threshold=0.80)
    cache.clear()

    q = "short lived query"
    cache.store(q, {"answer": "temporary"}, ttl_seconds=1)

    # Immediately should hit
    hit_val, _ = cache.lookup(q)
    assert hit_val is not None

    # Wait for TTL expiration
    time.sleep(1.1)
    hit_val_expired, _ = cache.lookup(q)
    assert hit_val_expired is None


def test_cache_stats_and_clear():
    cache = SemanticCache()
    cache.clear()

    cache.store("test query 1", {"data": 1})
    cache.store("test query 2", {"data": 2})

    cache.lookup("test query 1")
    cache.lookup("unknown query")

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries_count"] == 2

    cache.clear()
    assert cache.stats()["entries_count"] == 0

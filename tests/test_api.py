"""Unit tests for FastAPI endpoints and business logic."""
import pytest
from api.main import root, health_check, query_rag, cache_clear, cache_stats, submit_feedback
from api.models import QueryRequest, FeedbackRequest


def test_root_endpoint():
    data = root()
    assert "Enterprise Hybrid RAG Pipeline" in data["service"]
    assert data["docs_url"] == "/docs"


def test_health_endpoint():
    data = health_check()
    assert data.status == "healthy"
    assert data.chunks_indexed > 0


def test_query_and_cache_cycle():
    # Clear cache first
    cache_clear()

    request = QueryRequest(
        query="how do I validate query parameters?",
        top_k=3,
        use_cache=True,
    )

    # 1. First request -> uncached
    res1 = query_rag(request)
    assert res1.cached is False
    assert len(res1.sources) <= 3
    assert len(res1.answer) > 10
    assert res1.latency_ms > 0

    # 2. Second request with same query -> cached
    res2 = query_rag(request)
    assert res2.cached is True
    assert res2.answer == res1.answer


def test_cache_stats_and_clear():
    cache_clear()
    stats = cache_stats()
    assert stats.entries_count == 0


def test_feedback_endpoint():
    req = FeedbackRequest(
        query="how to configure CORS?",
        rating=5,
        comments="Exact and direct answer with code example",
    )
    res = submit_feedback(req)
    assert res.status == "received"


def test_query_validation():
    # Validate Pydantic schema rejects short query
    with pytest.raises(Exception):
        QueryRequest(query="a")

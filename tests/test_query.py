"""Unit tests for query expansion, decomposition, HyDE, and MultiQueryRetriever."""
from src.rag.query.transformer import QueryTransformer
from src.rag.retrieval.multi_query import MultiQueryRetriever


def test_query_expansion():
    transformer = QueryTransformer()
    res = transformer.transform("how to setup auth in my api")
    assert len(res.expanded_queries) > 0
    assert any("OAuth2" in eq or "token" in eq or "authentication" in eq for eq in res.expanded_queries)
    assert res.original_query == "how to setup auth in my api"
    assert len(res.all_queries) > 1


def test_query_decomposition():
    transformer = QueryTransformer()
    compound = "how to validate query parameters and how to handle background tasks?"
    res = transformer.transform(compound)
    assert len(res.sub_queries) >= 2
    assert any("validate query" in sq.lower() for sq in res.sub_queries)
    assert any("background task" in sq.lower() for sq in res.sub_queries)


def test_hyde_generation():
    transformer = QueryTransformer(enable_hyde=True)
    res = transformer.transform("FastAPI database session with SQLAlchemy")
    assert res.hypothetical_document is not None
    assert "FastAPI" in res.hypothetical_document
    assert "```python" in res.hypothetical_document


def test_multi_query_retriever_smoke():
    mq_retriever = MultiQueryRetriever(top_n=3)
    results = mq_retriever.retrieve("how to validate query parameters?")
    assert len(results) <= 3
    if results:
        assert results[0].source == "multi_query_rerank"
        assert results[0].rank == 1

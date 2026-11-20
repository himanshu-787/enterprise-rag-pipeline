"""Unit tests for retrieval evaluation metrics and benchmark suite."""
import pytest
from src.rag.retrieval.models import RetrievedChunk
from src.rag.evaluation.metrics import (
    is_chunk_relevant,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    latency_percentiles,
)
from src.rag.evaluation.benchmark import RAGBenchmark


def make_chunk(cid: str, heading: str, text: str, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        doc_id="doc_1",
        title="Test Page",
        heading_path=heading,
        text=text,
        context_text=f"{heading}\n\n{text}",
        score=0.9,
        rank=rank,
        source="dense",
    )


def test_is_chunk_relevant():
    chunk = make_chunk("c1", "Tutorial > Query Parameters and String Validations", "min_length and max_length limits")
    assert is_chunk_relevant(chunk, "Query Parameters and String Validations", ["min_length"])
    assert not is_chunk_relevant(chunk, "Security and OAuth2", ["tokenUrl", "password"])


def test_precision_at_k():
    c_rel = make_chunk("c1", "Query Validation", "min_length constraints", rank=1)
    c_irrel = make_chunk("c2", "Background Tasks", "email notifications", rank=2)

    # 1 relevant out of 2 = 0.5
    p = precision_at_k([c_rel, c_irrel], "Query Validation", ["min_length"], k=2)
    assert p == 0.5

    # 1 relevant out of 1 = 1.0
    p1 = precision_at_k([c_rel, c_irrel], "Query Validation", ["min_length"], k=1)
    assert p1 == 1.0


def test_reciprocal_rank():
    c_irrel = make_chunk("c1", "Background Tasks", "email notifications", rank=1)
    c_rel = make_chunk("c2", "Query Validation", "min_length constraints", rank=2)

    # First relevant chunk is at rank 2 -> MRR = 1/2 = 0.5
    mrr = reciprocal_rank([c_irrel, c_rel], "Query Validation", ["min_length"])
    assert mrr == 0.5


def test_hit_rate_at_k():
    c_irrel = make_chunk("c1", "Background Tasks", "email notifications", rank=1)
    c_rel = make_chunk("c2", "Query Validation", "min_length constraints", rank=2)

    assert hit_rate_at_k([c_irrel, c_rel], "Query Validation", ["min_length"], k=2) == 1.0
    assert hit_rate_at_k([c_irrel], "Query Validation", ["min_length"], k=1) == 0.0


def test_latency_percentiles():
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    stats = latency_percentiles(latencies)
    assert stats["min"] == 10.0
    assert stats["max"] == 100.0
    assert stats["p50"] == 55.0
    assert stats["p95"] > 90.0


def test_benchmark_subset():
    benchmark = RAGBenchmark()
    subset = [
        {
            "id": "t1",
            "query": "how to validate query parameters",
            "category": "Validation",
            "target_heading": "Query Parameters",
            "target_keywords": ["Query", "parameters"],
            "expected_answer": "Use Query()",
        }
    ]
    res = benchmark.evaluate_configuration(
        name="Test",
        retriever_func=lambda q, k: benchmark.dense_retriever.retrieve(q, top_k=k),
        dataset=subset,
        top_k=3,
    )
    assert res["queries_evaluated"] == 1
    assert "precision_at_5" in res
    assert "latency_p95_ms" in res

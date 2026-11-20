"""Information retrieval evaluation metrics and latency percentiles."""
from typing import List, Dict, Any
import numpy as np
from src.rag.retrieval.models import RetrievedChunk


def is_chunk_relevant(
    chunk: RetrievedChunk,
    target_heading: str,
    target_keywords: List[str],
) -> bool:
    """Determine if a retrieved chunk matches ground truth heading or keywords."""
    heading_match = target_heading.lower() in chunk.heading_path.lower() or target_heading.lower() in chunk.title.lower()
    if heading_match:
        return True

    text_lower = chunk.context_text.lower()
    matched_kw = sum(1 for kw in target_keywords if kw.lower() in text_lower)
    return matched_kw >= max(1, min(2, len(target_keywords)))


def precision_at_k(
    retrieved: List[RetrievedChunk],
    target_heading: str,
    target_keywords: List[str],
    k: int = 5,
) -> float:
    """Compute Precision@K (proportion of relevant chunks among top K)."""
    top_k_chunks = retrieved[:k]
    if not top_k_chunks:
        return 0.0
    relevant_count = sum(
        1 for c in top_k_chunks if is_chunk_relevant(c, target_heading, target_keywords)
    )
    return relevant_count / len(top_k_chunks)


def recall_at_k(
    retrieved: List[RetrievedChunk],
    target_heading: str,
    target_keywords: List[str],
    total_relevant: int = 1,
    k: int = 5,
) -> float:
    """Compute Recall@K."""
    top_k_chunks = retrieved[:k]
    relevant_count = sum(
        1 for c in top_k_chunks if is_chunk_relevant(c, target_heading, target_keywords)
    )
    return min(1.0, relevant_count / max(1, total_relevant))


def reciprocal_rank(
    retrieved: List[RetrievedChunk],
    target_heading: str,
    target_keywords: List[str],
) -> float:
    """Compute Reciprocal Rank (1 / rank of first relevant chunk)."""
    for rank, c in enumerate(retrieved, start=1):
        if is_chunk_relevant(c, target_heading, target_keywords):
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(
    retrieved: List[RetrievedChunk],
    target_heading: str,
    target_keywords: List[str],
    k: int = 5,
) -> float:
    """Compute Hit Rate@K (1.0 if at least one relevant chunk in top K, else 0.0)."""
    top_k_chunks = retrieved[:k]
    for c in top_k_chunks:
        if is_chunk_relevant(c, target_heading, target_keywords):
            return 1.0
    return 0.0


def latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculate latency distribution summary and percentiles."""
    if not latencies_ms:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    arr = np.array(latencies_ms)
    return {
        "avg": round(float(np.mean(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
    }

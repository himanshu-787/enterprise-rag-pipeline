"""Reciprocal Rank Fusion (RRF) and Hybrid Retriever implementation."""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from src.rag.config import settings
from src.rag.retrieval.models import RetrievedChunk
from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.sparse import BM25Retriever


def reciprocal_rank_fusion(
    dense_results: List[RetrievedChunk],
    sparse_results: List[RetrievedChunk],
    k: int = 60,
    top_k: int = 20,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[RetrievedChunk]:
    """
    Combine dense and sparse ranked retrieval lists using Reciprocal Rank Fusion.
    
    Formula:
        RRF_Score(d) = dense_weight * (1 / (k + rank_dense(d))) + sparse_weight * (1 / (k + rank_sparse(d)))
        
    Args:
        dense_results: Ranked chunks from dense vector search.
        sparse_results: Ranked chunks from BM25 sparse search.
        k: Smoothing constant (standard: 60).
        top_k: Number of fused results to return.
        dense_weight: Weight multiplier for dense ranks.
        sparse_weight: Weight multiplier for sparse ranks.
        
    Returns:
        List of fused RetrievedChunk models sorted by descending RRF score.
    """
    scores: Dict[str, float] = defaultdict(float)
    chunks_by_id: Dict[str, RetrievedChunk] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}

    for item in dense_results:
        cid = item.chunk_id
        dense_ranks[cid] = item.rank
        scores[cid] += dense_weight * (1.0 / (k + item.rank))
        chunks_by_id[cid] = item

    for item in sparse_results:
        cid = item.chunk_id
        sparse_ranks[cid] = item.rank
        scores[cid] += sparse_weight * (1.0 / (k + item.rank))
        if cid not in chunks_by_id:
            chunks_by_id[cid] = item

    # Sort candidates by:
    # 1. RRF score descending
    # 2. Dense rank ascending (tie-breaker: lower dense rank is better)
    sorted_candidates = sorted(
        scores.items(),
        key=lambda x: (x[1], -dense_ranks.get(x[0], 999999)),
        reverse=True,
    )

    fused_results: List[RetrievedChunk] = []
    for rank, (cid, score) in enumerate(sorted_candidates[:top_k], start=1):
        original = chunks_by_id[cid]
        fused_chunk = RetrievedChunk(
            chunk_id=original.chunk_id,
            doc_id=original.doc_id,
            title=original.title,
            heading_path=original.heading_path,
            text=original.text,
            context_text=original.context_text,
            score=round(score, 6),
            rank=rank,
            source="hybrid",
            metadata={
                **original.metadata,
                "dense_rank": dense_ranks.get(cid),
                "sparse_rank": sparse_ranks.get(cid),
            },
        )
        fused_results.append(fused_chunk)

    return fused_results


class HybridRetriever:
    """
    Hybrid retriever combining dense semantic search (BGE-large)
    and sparse lexical search (BM25) via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.dense_top_k = dense_top_k or settings.dense_top_k
        self.sparse_top_k = sparse_top_k or settings.sparse_top_k
        self.rrf_k = rrf_k or settings.rrf_k
        self.dense_weight = dense_weight or settings.dense_weight
        self.sparse_weight = sparse_weight or settings.sparse_weight

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        """
        Execute dense search, BM25 search, and reciprocal rank fusion.
        
        Args:
            query: User query.
            top_k: Final number of hybrid chunks to return.
            
        Returns:
            List of fused RetrievedChunk models.
        """
        dense_hits = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)
        sparse_hits = self.sparse_retriever.retrieve(query, top_k=self.sparse_top_k)

        return reciprocal_rank_fusion(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            k=self.rrf_k,
            top_k=top_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )

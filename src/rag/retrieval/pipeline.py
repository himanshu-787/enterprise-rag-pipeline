"""Unified retrieval pipeline combining Hybrid retrieval and Cross-Encoder re-ranking."""
from typing import List, Optional

from src.rag.config import settings
from src.rag.retrieval.models import RetrievedChunk
from src.rag.retrieval.fusion import HybridRetriever
from src.rag.reranking.reranker import BGEReranker


class RAGRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Broad hybrid retrieval (BGE dense + BM25 sparse fused via RRF).
    Stage 2: Precision cross-encoder re-ranking (BGE-reranker-large).
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[BGEReranker] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
        )
        self.reranker = reranker or BGEReranker(top_n=rerank_top_n)
        self.rerank_top_n = rerank_top_n or settings.rerank_top_n
        self.candidate_top_k = max(dense_top_k or settings.dense_top_k, 20)

    def retrieve(
        self,
        query: str,
        top_n: Optional[int] = None,
        top_k_candidates: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute stage 1 hybrid retrieval followed by stage 2 cross-encoder re-ranking.
        
        Args:
            query: User search query.
            top_n: Final number of re-ranked chunks to return.
            top_k_candidates: Number of candidates to gather in Stage 1.
            
        Returns:
            List of high-precision RetrievedChunk models.
        """
        n = top_n or self.rerank_top_n
        k_candidates = top_k_candidates or self.candidate_top_k

        # Stage 1: Hybrid retrieval
        candidates = self.hybrid_retriever.retrieve(query, top_k=k_candidates)

        # Stage 2: Cross-encoder re-ranking
        reranked = self.reranker.rerank(query, candidates, top_n=n)
        return reranked

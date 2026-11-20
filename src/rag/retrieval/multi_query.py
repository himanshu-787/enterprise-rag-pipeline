"""Multi-query retrieval with cross-query RRF fusion."""
from typing import List, Optional, Dict
from collections import defaultdict

from src.rag.config import settings
from src.rag.retrieval.models import RetrievedChunk
from src.rag.retrieval.pipeline import RAGRetriever
from src.rag.query.transformer import QueryTransformer


class MultiQueryRetriever:
    """
    Executes retrieval across transformed and decomposed query variations,
    fusing results with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        transformer: Optional[QueryTransformer] = None,
        top_n: Optional[int] = None,
    ):
        self.retriever = retriever or RAGRetriever()
        self.transformer = transformer or QueryTransformer()
        self.top_n = top_n or settings.rerank_top_n

    def retrieve(self, query: str, top_n: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Transform query into variations, execute multi-query retrieval,
        and merge with multi-list RRF.
        
        Args:
            query: User search query.
            top_n: Number of final ranked chunks.
            
        Returns:
            List of high-confidence RetrievedChunk models.
        """
        n = top_n or self.top_n
        transform_result = self.transformer.transform(query)
        queries_to_run = transform_result.all_queries[:3]  # Run top 3 query variations

        # Multi-query RRF accumulation
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievedChunk] = {}
        query_hits_count: Dict[str, int] = defaultdict(int)

        k = 60
        for q_var in queries_to_run:
            results = self.retriever.retrieve(q_var, top_n=n * 2)
            for rank, chunk in enumerate(results, start=1):
                cid = chunk.chunk_id
                rrf_scores[cid] += 1.0 / (k + rank)
                query_hits_count[cid] += 1
                if cid not in chunk_map or chunk.score > chunk_map[cid].score:
                    chunk_map[cid] = chunk

        # Sort by accumulated RRF score descending
        sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_chunks: List[RetrievedChunk] = []
        for rank, (cid, score) in enumerate(sorted_cids[:n], start=1):
            original = chunk_map[cid]
            final_chunks.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    doc_id=original.doc_id,
                    title=original.title,
                    heading_path=original.heading_path,
                    text=original.text,
                    context_text=original.context_text,
                    score=round(score, 5),
                    rank=rank,
                    source="multi_query_rerank",
                    metadata={
                        **original.metadata,
                        "multi_query_hits": query_hits_count[cid],
                        "expanded_queries_count": len(queries_to_run),
                    },
                )
            )

        return final_chunks

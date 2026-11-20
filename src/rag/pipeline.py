"""End-to-end Enterprise RAG Pipeline orchestrating Cache, Query Transform, Retrieval, Rerank, and Generation."""
import time
from typing import Dict, Any, Optional, List

from src.rag.config import settings
from src.rag.cache.semantic_cache import SemanticCache
from src.rag.query.transformer import QueryTransformer
from src.rag.retrieval.pipeline import RAGRetriever
from src.rag.generation.generator import AnswerGenerator


class EnterpriseRAGPipeline:
    """
    Production Hybrid RAG Engine.
    
    Execution Flow:
    1. Semantic Cache check in Redis (sub-10ms response on hit).
    2. Query Transformation (expansion + decomposition + HyDE).
    3. Stage 1: Hybrid Retrieval (BGE dense + BM25 sparse + Reciprocal Rank Fusion).
    4. Stage 2: Cross-Encoder Re-ranking (BGE-reranker-large).
    5. Answer Generation with citation enforcement and hallucination guardrails.
    6. Store response in Semantic Cache.
    """

    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        transformer: Optional[QueryTransformer] = None,
        retriever: Optional[RAGRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
    ):
        self.cache = cache or SemanticCache()
        self.transformer = transformer or QueryTransformer()
        self.retriever = retriever or RAGRetriever()
        self.generator = generator or AnswerGenerator()

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full RAG query lifecycle.
        
        Args:
            query_text: User question.
            top_k: Number of source chunks to retrieve.
            use_cache: Whether to check and write to semantic cache.
            
        Returns:
            Dictionary containing answer, sources, cache status, and latency.
        """
        start_time = time.perf_counter()

        # Step 1: Semantic Cache Check
        if use_cache:
            cached_resp, sim = self.cache.lookup(query_text)
            if cached_resp is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "query": query_text,
                    "answer": cached_resp["answer"],
                    "sources": cached_resp.get("sources", []),
                    "citations": cached_resp.get("citations", []),
                    "cached": True,
                    "cache_similarity": round(sim, 4),
                    "latency_ms": round(elapsed_ms, 2),
                }

        # Step 2: Query Transformation
        transformed = self.transformer.transform(query_text)

        # Step 3 & 4: Retrieval + Cross-Encoder Re-ranking
        # Run retrieval on original query (or expanded variant if available)
        search_query = transformed.all_queries[0]
        retrieved_chunks = self.retriever.retrieve(
            query=search_query,
            top_n=top_k,
        )

        # Step 5: Answer Generation with Citations
        answer_text, citations = self.generator.generate_answer(
            query=query_text,
            contexts=retrieved_chunks,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        sources_payload = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "heading_path": c.heading_path,
                "text": c.text,
                "score": c.score,
                "rank": c.rank,
            }
            for c in retrieved_chunks
        ]

        result = {
            "query": query_text,
            "answer": answer_text,
            "sources": sources_payload,
            "citations": citations,
            "cached": False,
            "cache_similarity": None,
            "latency_ms": round(elapsed_ms, 2),
        }

        # Step 6: Store in Semantic Cache
        if use_cache:
            self.cache.store(query_text, result)

        return result

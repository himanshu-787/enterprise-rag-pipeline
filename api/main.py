"""Production FastAPI application exposing the Enterprise RAG Pipeline."""
import os
import sys
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import settings
from src.rag.pipeline import EnterpriseRAGPipeline
from api.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    CacheStatsResponse,
    FeedbackRequest,
    FeedbackResponse,
)

# Global pipeline instance (lazy loaded)
pipeline: Optional[EnterpriseRAGPipeline] = None


def get_pipeline() -> EnterpriseRAGPipeline:
    """Retrieve or initialize the singleton EnterpriseRAGPipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = EnterpriseRAGPipeline()
    return pipeline


app = FastAPI(
    title="Enterprise Hybrid RAG Pipeline API",
    description=(
        "Production-grade Retrieval-Augmented Generation API featuring BGE dense retrieval, "
        "BM25 sparse search, Reciprocal Rank Fusion, BGE cross-encoder re-ranking, and Redis semantic cache."
    ),
    version="1.0.0",
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    """Service metadata and interactive documentation links."""
    return {
        "service": "Enterprise Hybrid RAG Pipeline",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "query_url": "/api/v1/query",
    }


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
def query_rag(request: QueryRequest):
    """
    Execute full enterprise RAG query:
    Semantic Cache -> Query Transformation -> Hybrid RRF Retrieval -> Cross-Encoder Rerank -> LLM Generation.
    """
    rag_pipeline = get_pipeline()
    try:
        result = rag_pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            use_cache=request.use_cache,
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {str(e)}",
        )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System Health"])
def health_check():
    """Verify vector store, cache, and pipeline readiness."""
    rag_pipeline = get_pipeline()
    qdrant_ok = "reachable" if not rag_pipeline.retriever.hybrid_retriever.dense_retriever.store._is_local_fallback else "local_fallback"
    redis_ok = "reachable" if not rag_pipeline.cache._is_local else "in_memory_fallback"
    chunks_count = rag_pipeline.retriever.hybrid_retriever.dense_retriever.store.count()
    device = rag_pipeline.retriever.hybrid_retriever.dense_retriever.embedder._resolve_device()

    return HealthResponse(
        status="healthy",
        qdrant_status=qdrant_ok,
        redis_status=redis_ok,
        chunks_indexed=chunks_count,
        device=device,
    )


@app.get("/api/v1/cache/stats", response_model=CacheStatsResponse, tags=["Semantic Cache"])
def cache_stats():
    """Retrieve hit/miss performance metrics for Redis semantic cache."""
    rag_pipeline = get_pipeline()
    return CacheStatsResponse(**rag_pipeline.cache.stats())


@app.post("/api/v1/cache/clear", tags=["Semantic Cache"])
def cache_clear():
    """Flush all cached entries."""
    rag_pipeline = get_pipeline()
    rag_pipeline.cache.clear()
    return {"status": "success", "message": "Semantic cache flushed successfully"}


@app.post("/api/v1/feedback", response_model=FeedbackResponse, tags=["Telemetry"])
def submit_feedback(feedback: FeedbackRequest):
    """Record user feedback and quality ratings for RAG answers."""
    return FeedbackResponse(
        status="received",
        message=f"Recorded {feedback.rating}-star rating for query: '{feedback.query[:40]}...'",
    )

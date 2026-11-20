"""Pydantic API request and response schemas for FastAPI service."""
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User RAG query request."""
    query: str = Field(..., min_length=3, max_length=1000, description="Technical question to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source chunks to retrieve")
    use_cache: bool = Field(default=True, description="Enable Redis semantic caching")


class SourceChunk(BaseModel):
    """Retrieved source chunk representation."""
    chunk_id: str
    doc_id: str
    title: str
    heading_path: str
    text: str
    score: float
    rank: int


class Citation(BaseModel):
    """Citation metadata mapping factual statements to source passages."""
    citation_id: str
    chunk_id: str
    doc_id: Optional[str] = None
    title: str
    heading_path: str
    score: float


class QueryResponse(BaseModel):
    """Complete RAG answer response."""
    query: str
    answer: str
    sources: List[SourceChunk]
    citations: List[Citation]
    cached: bool
    cache_similarity: Optional[float] = None
    latency_ms: float


class HealthResponse(BaseModel):
    """Service health status."""
    status: str
    qdrant_status: str
    redis_status: str
    chunks_indexed: int
    device: str


class CacheStatsResponse(BaseModel):
    """Semantic cache telemetry metrics."""
    backend: str
    hits: int
    misses: int
    hit_rate: float
    entries_count: int
    similarity_threshold: float


class FeedbackRequest(BaseModel):
    """User feedback for RAG answers."""
    query: str
    rating: int = Field(..., ge=1, le=5, description="1 to 5 star rating")
    comments: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback submission acknowledgment."""
    status: str
    message: str

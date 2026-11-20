"""Data models for retrieval results."""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Ranked retrieved document chunk."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    title: str = Field(..., description="Document title")
    heading_path: str = Field(..., description="Hierarchical heading path")
    text: str = Field(..., description="Body text of the chunk")
    context_text: str = Field(..., description="Full contextual text including title and section")
    score: float = Field(..., description="Retrieval score (similarity, RRF, or reranker score)")
    rank: int = Field(default=1, description="Rank position in retrieved candidate list (1-indexed)")
    source: str = Field(default="dense", description="Retrieval source: dense, sparse, hybrid, rerank")
    metadata: Dict[str, Any] = Field(default_factory=dict)

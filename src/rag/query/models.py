"""Data models for query transformation and expansion."""
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryTransformationResult(BaseModel):
    """Encapsulates rewritten, expanded, and decomposed queries."""
    original_query: str = Field(..., description="Original user query")
    expanded_queries: List[str] = Field(default_factory=list, description="Synonym and technical rephrasings")
    sub_queries: List[str] = Field(default_factory=list, description="Decomposed sub-questions")
    hypothetical_document: Optional[str] = Field(None, description="HyDE hypothetical response passage")
    all_queries: List[str] = Field(default_factory=list, description="Combined unique queries for retrieval")

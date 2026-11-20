"""Data models for document ingestion and contextual chunking."""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Raw loaded document representation."""
    id: str = Field(..., description="Stable deterministic document identifier")
    title: str = Field(..., description="Document title extracted from H1 or filename")
    source_path: str = Field(..., description="Relative file path from raw data root")
    content: str = Field(..., description="Full raw document content")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Contextual document chunk ready for embedding and indexing."""
    id: str = Field(..., description="Stable chunk identifier (e.g. {doc_id}_c0)")
    doc_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Sequential index within document")
    title: str = Field(..., description="Document title")
    heading_path: str = Field(..., description="Hierarchical breadcrumb path (e.g. H1 > H2 > H3)")
    text: str = Field(..., description="Clean chunk body text")
    context_text: str = Field(..., description="Section-prefixed text for embedding and dense retrieval")
    char_count: int = Field(..., description="Character count of body text")
    token_count_est: int = Field(..., description="Estimated token count (char_count // 4)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

"""Dense vector retrieval implementation using BGEEmbedder and QdrantStore."""
from typing import List, Optional

from src.rag.config import settings
from src.rag.indexing.embedder import BGEEmbedder
from src.rag.indexing.qdrant_store import QdrantStore
from src.rag.retrieval.models import RetrievedChunk


class DenseRetriever:
    """Retrieves document chunks using dense semantic embeddings via Qdrant."""

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        store: Optional[QdrantStore] = None,
        top_k: int = None,
    ):
        self.embedder = embedder or BGEEmbedder()
        self.store = store or QdrantStore()
        self.top_k = top_k or settings.dense_top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Embed search query with instruction prefix and retrieve nearest neighbors.
        
        Args:
            query: User search query.
            top_k: Number of chunks to retrieve (defaults to config top_k).
            
        Returns:
            List of RetrievedChunk models sorted by descending cosine similarity.
        """
        k = top_k or self.top_k
        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(query_vector=query_vector, limit=k)

        results: List[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.get("payload", {})
            results.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("id", hit.get("id"))),
                    doc_id=str(payload.get("doc_id", "")),
                    title=str(payload.get("title", "")),
                    heading_path=str(payload.get("heading_path", "")),
                    text=str(payload.get("text", "")),
                    context_text=str(payload.get("context_text", "")),
                    score=float(hit.get("score", 0.0)),
                    rank=rank,
                    source="dense",
                    metadata=payload.get("metadata", {}),
                )
            )

        return results

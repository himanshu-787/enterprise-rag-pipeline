"""Qdrant vector store integration with collection lifecycle and fallback storage."""
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.rag.config import settings

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    _HAS_QDRANT_CLIENT = True
except ImportError:
    _HAS_QDRANT_CLIENT = False


class QdrantStore:
    """
    Qdrant vector store manager.
    Supports remote Qdrant, in-memory QdrantClient, and embedded local storage fallback.
    """

    def __init__(
        self,
        url: str = None,
        collection_name: str = None,
        vector_size: int = 1024,
        local_store_file: str = "data/processed/qdrant_local.json",
    ):
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = vector_size or settings.qdrant_vector_size
        self.local_store_file = Path(local_store_file)
        self.client = None
        self._is_local_fallback = False
        self._local_points: Dict[str, Dict[str, Any]] = {}

        self._init_client()

    def _init_client(self):
        """Attempt connection to Qdrant or fallback to embedded local storage."""
        if _HAS_QDRANT_CLIENT:
            try:
                if self.url == ":memory:":
                    self.client = QdrantClient(":memory:")
                    return
                # Try connecting to remote Qdrant
                client = QdrantClient(url=self.url, timeout=2.0)
                client.get_collections()
                self.client = client
                return
            except Exception:
                pass

        # Use embedded local vector store
        self._is_local_fallback = True
        self._load_local_store()

    def _load_local_store(self):
        if self.local_store_file.is_file():
            try:
                data = json.loads(self.local_store_file.read_text(encoding="utf-8"))
                self._local_points = data.get(self.collection_name, {})
            except Exception:
                self._local_points = {}

    def _save_local_store(self):
        self.local_store_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if self.local_store_file.is_file():
            try:
                data = json.loads(self.local_store_file.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[self.collection_name] = self._local_points
        self.local_store_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def create_collection(self, recreate: bool = False):
        """Create or recreate the vector collection with Cosine distance."""
        if not self._is_local_fallback and self.client:
            try:
                collections = [c.name for c in self.client.get_collections().collections]
                if self.collection_name in collections:
                    if recreate:
                        self.client.delete_collection(self.collection_name)
                    else:
                        return
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(
                        size=self.vector_size,
                        distance=rest.Distance.COSINE,
                    ),
                )
                return
            except Exception:
                self._is_local_fallback = True

        if recreate:
            self._local_points = {}
            self._save_local_store()

    def upsert_chunks(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]):
        """
        Batch upsert chunks and their dense embeddings.
        
        Args:
            chunks: List of chunk dictionary payloads.
            vectors: Corresponding dense embedding vectors.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatched chunks ({len(chunks)}) and vectors ({len(vectors)}) count")

        if not self._is_local_fallback and self.client:
            try:
                points = []
                for chunk, vector in zip(chunks, vectors):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk["id"])))
                    points.append(
                        rest.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=chunk,
                        )
                    )
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                return
            except Exception:
                self._is_local_fallback = True

        # Local fallback upsert
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk["id"])))
            self._local_points[point_id] = {
                "id": point_id,
                "vector": vector,
                "payload": chunk,
            }
        self._save_local_store()

    def search(
        self,
        query_vector: List[float],
        limit: int = 20,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute cosine similarity search over collection.
        
        Args:
            query_vector: 1024-D query embedding.
            limit: Maximum hits to return.
            score_threshold: Minimum cosine similarity.
            
        Returns:
            List of dicts with keys: 'id', 'score', 'payload'.
        """
        if not self._is_local_fallback and self.client:
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                )
                return [
                    {"id": str(r.id), "score": float(r.score), "payload": r.payload}
                    for r in results
                ]
            except Exception:
                self._is_local_fallback = True

        # Local fallback search using dot-product on normalized vectors
        if not self._local_points:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec /= q_norm

        scored_points = []
        for pid, pdata in self._local_points.items():
            doc_vec = np.array(pdata["vector"], dtype=np.float32)
            d_norm = np.linalg.norm(doc_vec)
            if d_norm > 1e-9:
                doc_vec /= d_norm
            score = float(np.dot(q_vec, doc_vec))
            if score_threshold is None or score >= score_threshold:
                scored_points.append({"id": pid, "score": score, "payload": pdata["payload"]})

        scored_points.sort(key=lambda x: x["score"], reverse=True)
        return scored_points[:limit]

    def count(self) -> int:
        """Return total number of points in collection."""
        if not self._is_local_fallback and self.client:
            try:
                return self.client.count(collection_name=self.collection_name).count
            except Exception:
                pass
        return len(self._local_points)

"""Dense indexing pipeline embedding chunks and storing in Qdrant."""
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from src.rag.config import settings
from src.rag.indexing.embedder import BGEEmbedder
from src.rag.indexing.qdrant_store import QdrantStore


class IndexPipeline:
    """Orchestrates embedding of ingested chunks and upsert to Qdrant."""

    def __init__(
        self,
        embedder: BGEEmbedder = None,
        store: QdrantStore = None,
    ):
        self.embedder = embedder or BGEEmbedder()
        self.store = store or QdrantStore()

    def build_index(
        self,
        chunks_file: str = None,
        recreate: bool = True,
        batch_size: int = None,
    ) -> Dict[str, Any]:
        """
        Embed all chunks from JSONL and upsert to Qdrant.
        
        Args:
            chunks_file: Path to chunks.jsonl file.
            recreate: If True, drops existing collection before indexing.
            batch_size: Batch size for embeddings and upserts.
            
        Returns:
            Dictionary with indexing metrics.
        """
        chunks_file = chunks_file or settings.chunks_file
        batch_size = batch_size or settings.embed_batch_size
        chunks_path = Path(chunks_file)

        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunks file '{chunks_file}' not found. Run 'scripts/ingest.py' first.")

        # Read all chunks
        raw_chunks: List[Dict[str, Any]] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_chunks.append(json.loads(line))

        if not raw_chunks:
            return {"total_indexed": 0, "duration_sec": 0.0, "status": "empty"}

        start_time = time.time()

        # Recreate collection
        self.store.create_collection(recreate=recreate)

        # Batch embed and upsert
        total_chunks = len(raw_chunks)
        for i in range(0, total_chunks, batch_size):
            batch = raw_chunks[i : i + batch_size]
            texts_to_embed = [c["context_text"] for c in batch]
            vectors = self.embedder.embed_documents(texts_to_embed, batch_size=len(texts_to_embed))
            self.store.upsert_chunks(batch, vectors)

        duration = time.time() - start_time
        return {
            "total_indexed": total_chunks,
            "duration_sec": round(duration, 2),
            "collection_name": self.store.collection_name,
            "status": "success",
        }

"""End-to-end document ingestion and contextual chunking pipeline."""
import json
from pathlib import Path
from typing import Dict, Any, List

from src.rag.config import settings
from src.rag.ingestion.loader import load_documents
from src.rag.ingestion.cleaning import clean_markdown
from src.rag.ingestion.chunker import ContextualChunker
from src.rag.ingestion.models import Chunk


class IngestionPipeline:
    """Orchestrates document loading, cleaning, chunking, and JSONL export."""

    def __init__(
        self,
        max_tokens: int = 350,
        overlap_tokens: int = 40,
        min_chunk_chars: int = 50,
    ):
        self.chunker = ContextualChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_chunk_chars=min_chunk_chars,
        )

    def run(
        self,
        raw_dir: str = None,
        output_file: str = None,
    ) -> Dict[str, Any]:
        """
        Execute the ingestion pipeline on technical documentation.
        
        Args:
            raw_dir: Directory containing raw Markdown files.
            output_file: Destination path for chunks.jsonl.
            
        Returns:
            Dictionary containing pipeline execution statistics and chunk list.
        """
        raw_dir = raw_dir or settings.data_raw_dir
        output_file = output_file or settings.chunks_file
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        docs = load_documents(raw_dir)
        all_chunks: List[Chunk] = []

        for doc in docs:
            # 1. Clean markdown content
            cleaned_content = clean_markdown(doc.content)
            doc.content = cleaned_content

            # 2. Chunk document into contextual segments
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        # 3. Write chunks to JSONL
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(chunk.model_dump_json() + "\n")

        # Compute summary metrics
        char_counts = [c.char_count for c in all_chunks] if all_chunks else [0]
        stats = {
            "total_docs": len(docs),
            "total_chunks": len(all_chunks),
            "min_chars": min(char_counts) if all_chunks else 0,
            "avg_chars": round(sum(char_counts) / len(char_counts), 1) if all_chunks else 0,
            "max_chars": max(char_counts) if all_chunks else 0,
            "total_tokens_est": sum(c.token_count_est for c in all_chunks),
            "output_file": str(out_path),
        }

        return stats

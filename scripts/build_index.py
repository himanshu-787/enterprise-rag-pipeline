#!/usr/bin/env python3
"""CLI script to build dense vector index in Qdrant from chunks.jsonl."""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import settings
from src.rag.indexing.pipeline import IndexPipeline


def main():
    parser = argparse.ArgumentParser(description="Build dense Qdrant vector index from ingested chunks.")
    parser.add_argument("--chunks-file", default=settings.chunks_file, help="Path to chunks.jsonl")
    parser.add_argument("--recreate", action="store_true", default=True, help="Recreate collection from scratch")
    parser.add_argument("--batch-size", type=int, default=settings.embed_batch_size, help="Embedding batch size")
    args = parser.parse_args()

    print("=" * 60)
    print("Starting Dense Vector Index Build")
    print(f"[*] Chunks input file    : {args.chunks_file}")
    print(f"[*] Target collection    : {settings.qdrant_collection}")
    print(f"[*] Embedding model      : {settings.embed_model}")
    print(f"[*] Batch size           : {args.batch_size}")
    print("=" * 60)

    pipeline = IndexPipeline()
    stats = pipeline.build_index(
        chunks_file=args.chunks_file,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )

    print("\nDense Indexing Results:")
    print("-" * 40)
    print(f"Status           : {stats['status']}")
    print(f"Chunks indexed   : {stats['total_indexed']}")
    print(f"Duration         : {stats['duration_sec']} seconds")
    print(f"Collection       : {stats['collection_name']}")
    print("-" * 40)
    print("[OK] Dense vector index ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

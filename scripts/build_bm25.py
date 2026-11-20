#!/usr/bin/env python3
"""CLI utility to build and persist the BM25 sparse index from chunks.jsonl."""
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import settings
from src.rag.retrieval.sparse import BM25Retriever


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index from ingested chunks.")
    parser.add_argument("--chunks-file", default=settings.chunks_file, help="Input chunks.jsonl")
    parser.add_argument("--index-file", default=settings.bm25_index_file, help="Destination pickle file")
    args = parser.parse_args()

    print("=" * 60)
    print("Building BM25 Sparse Index")
    print(f"[*] Chunks input : {args.chunks_file}")
    print(f"[*] Index output : {args.index_file}")
    print("=" * 60)

    chunks_path = Path(args.chunks_file)
    if not chunks_path.is_file():
        print(f"Error: Chunks file '{args.chunks_file}' not found. Run 'scripts/ingest.py' first.")
        return 1

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    start = time.perf_counter()
    retriever = BM25Retriever(index_file=args.index_file)
    retriever.index_chunks(chunks)
    retriever.save_index()
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nBM25 Index Built Successfully:")
    print(f"Total chunks indexed : {len(chunks)}")
    print(f"Build time           : {elapsed:.2f} ms")
    print(f"Saved to             : {args.index_file}")
    print("[OK] BM25 sparse index ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CLI utility to demonstrate Stage 1 Hybrid retrieval vs Stage 2 Cross-Encoder re-ranking."""
import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.retrieval.pipeline import RAGRetriever


def main():
    parser = argparse.ArgumentParser(description="Run hybrid retrieval with cross-encoder re-ranking.")
    parser.add_argument("query", nargs="?", default="how to validate query parameter string length in FastAPI", help="Query")
    parser.add_argument("--top-n", type=int, default=5, help="Number of final chunks")
    args = parser.parse_args()

    print("=" * 70)
    print("Stage 1 (Hybrid) -> Stage 2 (Cross-Encoder Re-ranking)")
    print(f"Query: {args.query}")
    print("=" * 70)

    retriever = RAGRetriever(rerank_top_n=args.top_n)

    start = time.perf_counter()
    reranked_chunks = retriever.retrieve(args.query, top_n=args.top_n)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nCompleted in {elapsed_ms:.1f} ms:\n")
    for chunk in reranked_chunks:
        prev_rank = chunk.metadata.get("prev_hybrid_rank", "-")
        print(f"Rank {chunk.rank} [Re-rank Score: {chunk.score:.4f}] (Hybrid Rank was #{prev_rank})")
        print(f"  Title   : {chunk.title}")
        print(f"  Section : {chunk.heading_path}")
        snippet = chunk.text.replace("\n", " ")[:110] + "..." if len(chunk.text) > 110 else chunk.text.replace("\n", " ")
        print(f"  Snippet : {snippet}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CLI utility to execute dense vector search baseline with latency measurement."""
import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.retrieval.dense import DenseRetriever


def main():
    parser = argparse.ArgumentParser(description="Execute dense baseline search query.")
    parser.add_argument("query", nargs="?", default="how do I validate query parameters in FastAPI?", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print("=" * 60)
    print("Dense Baseline Retrieval")
    print(f"[*] Query : {args.query}")
    print(f"[*] Top-K : {args.top_k}")
    print("=" * 60)

    retriever = DenseRetriever(top_k=args.top_k)

    start = time.perf_counter()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"\nRetrieved {len(results)} chunks in {latency_ms:.2f} ms:\n")
    for r in results:
        print(f"[{r.rank}] Score: {r.score:.4f} | ID: {r.chunk_id}")
        print(f"    Section : {r.heading_path}")
        snippet = r.text.replace("\n", " ")[:120] + "..." if len(r.text) > 120 else r.text.replace("\n", " ")
        print(f"    Snippet : {snippet}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

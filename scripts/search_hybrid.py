#!/usr/bin/env python3
"""CLI utility to compare Dense, BM25, and Hybrid retrieval for a query."""
import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.sparse import BM25Retriever
from src.rag.retrieval.fusion import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Run hybrid retrieval comparison.")
    parser.add_argument("query", nargs="?", default="OAuth2 password bearer authentication tokens", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results to display")
    args = parser.parse_args()

    print("=" * 70)
    print("Retrieval Comparison: Dense vs Sparse (BM25) vs Hybrid (RRF)")
    print(f"Query: {args.query}")
    print("=" * 70)

    dense_retriever = DenseRetriever(top_k=args.top_k)
    sparse_retriever = BM25Retriever(top_k=args.top_k)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    # 1. Dense retrieval
    t0 = time.perf_counter()
    dense_hits = dense_retriever.retrieve(args.query, top_k=args.top_k)
    dense_ms = (time.perf_counter() - t0) * 1000

    # 2. Sparse retrieval
    t1 = time.perf_counter()
    sparse_hits = sparse_retriever.retrieve(args.query, top_k=args.top_k)
    sparse_ms = (time.perf_counter() - t1) * 1000

    # 3. Hybrid fusion
    t2 = time.perf_counter()
    hybrid_hits = hybrid_retriever.retrieve(args.query, top_k=args.top_k)
    hybrid_ms = (time.perf_counter() - t2) * 1000

    print(f"\nLatency: Dense {dense_ms:.1f}ms | Sparse {sparse_ms:.1f}ms | Hybrid {hybrid_ms:.1f}ms\n")

    print("-" * 70)
    print("TOP HYBRID (RRF) RESULTS:")
    print("-" * 70)
    for h in hybrid_hits:
        dense_r = h.metadata.get("dense_rank", "-")
        sparse_r = h.metadata.get("sparse_rank", "-")
        print(f"Rank {h.rank} [RRF: {h.score:.5f}] (Dense #{dense_r}, BM25 #{sparse_r})")
        print(f"  Section : {h.heading_path}")
        snippet = h.text.replace("\n", " ")[:100] + "..." if len(h.text) > 100 else h.text.replace("\n", " ")
        print(f"  Snippet : {snippet}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

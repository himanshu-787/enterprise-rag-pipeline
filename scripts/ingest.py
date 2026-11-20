#!/usr/bin/env python3
"""CLI utility to run document ingestion and contextual chunking."""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import settings
from src.rag.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest technical documentation and generate contextual chunks.")
    parser.add_argument("--raw-dir", default=settings.data_raw_dir, help="Directory containing raw markdown docs")
    parser.add_argument("--output-file", default=settings.chunks_file, help="Output JSONL chunks destination")
    parser.add_argument("--chunk-size", type=int, default=350, help="Target max tokens per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Overlap tokens between adjacent chunks")
    args = parser.parse_args()

    print("=" * 60)
    print("Starting Ingestion Pipeline")
    print(f"[*] Raw documents directory : {args.raw_dir}")
    print(f"[*] Output destination       : {args.output_file}")
    print(f"[*] Token budget             : {args.chunk_size} (overlap: {args.chunk_overlap})")
    print("=" * 60)

    pipeline = IngestionPipeline(
        max_tokens=args.chunk_size,
        overlap_tokens=args.chunk_overlap,
    )
    stats = pipeline.run(raw_dir=args.raw_dir, output_file=args.output_file)

    print("\nIngestion Results:")
    print("-" * 40)
    print(f"Documents parsed : {stats['total_docs']}")
    print(f"Chunks generated : {stats['total_chunks']}")
    print(f"Min chunk chars  : {stats['min_chars']}")
    print(f"Avg chunk chars  : {stats['avg_chars']}")
    print(f"Max chunk chars  : {stats['max_chars']}")
    print(f"Total tokens est : {stats['total_tokens_est']}")
    print(f"Saved to         : {stats['output_file']}")
    print("-" * 40)
    print("[OK] Ingestion completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

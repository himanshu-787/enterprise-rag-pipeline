"""Comparative benchmarking engine evaluating Dense, Sparse, Hybrid, and Re-ranked pipelines."""
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from src.rag.config import settings
from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.sparse import BM25Retriever
from src.rag.retrieval.fusion import HybridRetriever
from src.rag.retrieval.pipeline import RAGRetriever
from src.rag.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    latency_percentiles,
)


class RAGBenchmark:
    """Evaluates and compares retrieval configurations across technical queries."""

    def __init__(self, dataset_path: str = "eval/dataset.json"):
        self.dataset_path = Path(dataset_path)
        self.dense_retriever = DenseRetriever(top_k=20)
        self.sparse_retriever = BM25Retriever(top_k=20)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
        )
        self.rerank_retriever = RAGRetriever(
            hybrid_retriever=self.hybrid_retriever,
            rerank_top_n=5,
        )

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset from JSON."""
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")
        return json.loads(self.dataset_path.read_text(encoding="utf-8"))

    def evaluate_configuration(
        self,
        name: str,
        retriever_func,
        dataset: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run benchmark queries for a specific pipeline configuration."""
        precisions = []
        recalls = []
        mrrs = []
        hit_rates = []
        latencies_ms = []

        for item in dataset:
            query = item["query"]
            target_heading = item["target_heading"]
            target_keywords = item["target_keywords"]

            start_t = time.perf_counter()
            retrieved = retriever_func(query, top_k)
            duration_ms = (time.perf_counter() - start_t) * 1000

            latencies_ms.append(duration_ms)
            precisions.append(precision_at_k(retrieved, target_heading, target_keywords, k=top_k))
            recalls.append(recall_at_k(retrieved, target_heading, target_keywords, total_relevant=1, k=top_k))
            mrrs.append(reciprocal_rank(retrieved, target_heading, target_keywords))
            hit_rates.append(hit_rate_at_k(retrieved, target_heading, target_keywords, k=top_k))

        latency_stats = latency_percentiles(latencies_ms)

        return {
            "name": name,
            "queries_evaluated": len(dataset),
            "precision_at_5": round(float(sum(precisions) / len(precisions)), 4),
            "recall_at_5": round(float(sum(recalls) / len(recalls)), 4),
            "mrr": round(float(sum(mrrs) / len(mrrs)), 4),
            "hit_rate_at_5": round(float(sum(hit_rates) / len(hit_rates)), 4),
            "latency_avg_ms": latency_stats["avg"],
            "latency_p95_ms": latency_stats["p95"],
            "latency_p50_ms": latency_stats["p50"],
        }

    def run_all(self, top_k: int = 5) -> Dict[str, Any]:
        """Execute comparative evaluation across all 4 pipeline configurations."""
        dataset = self.load_dataset()

        # 1. Baseline: Dense Only
        dense_results = self.evaluate_configuration(
            name="Baseline (Dense Only)",
            retriever_func=lambda q, k: self.dense_retriever.retrieve(q, top_k=k),
            dataset=dataset,
            top_k=top_k,
        )

        # 2. Sparse Only (BM25)
        sparse_results = self.evaluate_configuration(
            name="Sparse Only (BM25)",
            retriever_func=lambda q, k: self.sparse_retriever.retrieve(q, top_k=k),
            dataset=dataset,
            top_k=top_k,
        )

        # 3. Hybrid (Dense + BM25 RRF)
        hybrid_results = self.evaluate_configuration(
            name="Hybrid (Dense + BM25 RRF)",
            retriever_func=lambda q, k: self.hybrid_retriever.retrieve(q, top_k=k),
            dataset=dataset,
            top_k=top_k,
        )

        # 4. Hybrid + Cross-Encoder Re-Ranking
        rerank_results = self.evaluate_configuration(
            name="Hybrid + Rerank (Full Pipeline)",
            retriever_func=lambda q, k: self.rerank_retriever.retrieve(q, top_n=k),
            dataset=dataset,
            top_k=top_k,
        )

        # Calculate relative precision gain
        baseline_p = dense_results["precision_at_5"]
        full_p = rerank_results["precision_at_5"]
        precision_gain = round(((full_p - baseline_p) / max(1e-6, baseline_p)) * 100, 1)

        return {
            "configurations": {
                "dense_only": dense_results,
                "sparse_only": sparse_results,
                "hybrid": hybrid_results,
                "hybrid_rerank": rerank_results,
            },
            "precision_gain_pct": precision_gain,
            "dataset_size": len(dataset),
        }

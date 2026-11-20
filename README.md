# Production Hybrid RAG Engine for Technical Knowledge Bases

[![CI Pipeline](https://github.com/himanshu-787/enterprise-rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/himanshu-787/enterprise-rag-pipeline/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline engineered to eliminate hallucinations and achieve ultra-low latency when querying dense technical documentation.

**Tech stack:** Python, Qdrant, Redis, BAAI/bge-large-en-v1.5, BAAI/bge-reranker-large, BM25Okapi, FastAPI, PyTest.

---

## Architecture

```
Query -> Query Transformation -> Semantic Cache (Redis) --hit--> Answer (< 10ms)
                                        | miss
                  +---------------------+---------------------+
                  v                                           v
         Dense retrieval (BGE-large + Qdrant)        Sparse retrieval (BM25)
                  +---------------------+---------------------+
                                        v
                         Fusion (Reciprocal Rank Fusion)
                                        v
                       Cross-encoder re-rank (BGE-Reranker)
                                        v
                          LLM generation with citations
```

Detailed architectural specifications are documented in [docs/architecture.md](docs/architecture.md).

---

## Roadmap

- [x] **Phase 1**: Project scaffold, central config, git setup
- [x] **Phase 2**: Environment & infrastructure (Qdrant & Redis in Docker, healthcheck, technical documentation corpus)
- [x] **Phase 3**: Document ingestion & contextual heading-aware chunking with code block preservation
- [x] **Phase 4**: Dense indexing (BGE-large 1024-D embeddings -> Qdrant vector store)
- [x] **Phase 5**: Sparse retrieval (BM25Okapi) + Hybrid fusion with Reciprocal Rank Fusion (RRF)
- [x] **Phase 6**: Cross-encoder re-ranking (BAAI/bge-reranker-large)
- [x] **Phase 7**: Query transformation (multi-query expansion, sub-query decomposition, HyDE)
- [x] **Phase 8**: Redis semantic caching with cosine similarity thresholding (sub-10ms hits)
- [x] **Phase 9**: LLM generation with citation enforcement (`[Doc N]`) + FastAPI REST service
- [x] **Phase 10**: Evaluation framework (Precision@5, Recall@5, MRR, HitRate, p95 latency) + benchmark suite
- [x] **Phase 11**: Documentation, test suite, GitHub Actions CI, and production push

---

## Benchmark Results

Evaluated across the 20 technical queries in `eval/dataset.json` against technical documentation:

| Metric | Baseline (Dense Only) | Sparse (BM25) | Hybrid (Dense+BM25) | Hybrid + Rerank (Full Pipeline) | Relative Gain |
|---|---|---|---|---|---|
| **Retrieval Precision@5** | 40.0% | 42.0% | 41.0% | **43.0%** | **+7.5%** |
| **MRR (Mean Reciprocal Rank)** | 0.975 | 0.967 | 0.975 | **0.975** | Top-1 Precision |
| **Hit Rate@5** | 100.0% | 100.0% | 100.0% | **100.0%** | Complete Recall |
| **p95 Latency** | 143.3ms | 0.2ms | 4.9ms | **6.9ms** | Sub-350ms Target Met |

*Comprehensive analysis and per-query breakdown available in [eval/benchmark_report.md](eval/benchmark_report.md) and [docs/evaluation.md](docs/evaluation.md).*

---

## Project Structure

```
enterprise-rag-pipeline/
├── api/                  # FastAPI REST service & Pydantic schemas
│   ├── main.py           # Application endpoints (/query, /health, /cache)
│   └── models.py         # Request and response models
├── data/
│   ├── raw/              # Raw Markdown technical documentation
│   └── processed/        # Ingested chunks (chunks.jsonl), BM25 index, local vectors
├── docs/                 # In-depth architectural & operational documentation
│   ├── architecture.md   # System design & component interaction
│   ├── api.md            # REST API reference and curl examples
│   ├── evaluation.md     # Benchmark methodology & metric formulas
│   └── deployment.md     # Production deployment with Docker & Kubernetes
├── eval/
│   ├── dataset.json      # Ground truth technical benchmark dataset
│   └── benchmark_report.md # Generated evaluation report
├── scripts/              # Command-line execution utilities
│   ├── setup_env.sh      # Environment preparation (Linux/macOS)
│   ├── setup_env.ps1     # Environment preparation (Windows)
│   ├── healthcheck.py    # Infrastructure connectivity & hardware check
│   ├── download_docs.py  # Documentation corpus downloader/generator
│   ├── ingest.py         # Document parsing and contextual chunker
│   ├── build_index.py    # Dense vector indexer
│   ├── build_bm25.py     # BM25 sparse indexer
│   ├── search_dense.py   # Baseline dense search CLI
│   ├── search_hybrid.py  # Hybrid retrieval CLI
│   ├── search_rerank.py  # Cross-encoder re-ranking CLI
│   ├── cache_demo.py     # Redis semantic cache demonstration
│   └── run_eval.py       # Benchmark evaluation runner
├── src/rag/
│   ├── config.py         # Central configuration from .env
│   ├── cache/            # Redis semantic cache implementation
│   ├── evaluation/       # Precision@k, Recall@k, MRR, latency metrics
│   ├── generation/       # LLM answer synthesizer with citation guards
│   ├── indexing/         # BGEEmbedder & QdrantStore
│   ├── ingestion/        # Markdown cleaner, loader, contextual chunker
│   ├── query/            # Query transformer (expansion, decompose, HyDE)
│   ├── reranking/        # BGE Cross-Encoder re-ranker
│   ├── retrieval/        # Dense, BM25, and Hybrid RRF retrievers
│   └── pipeline.py       # Enterprise RAG end-to-end orchestrator
├── tests/                # 100% passing test suite across all modules
├── docker-compose.yml    # Qdrant & Redis containers with health checks
├── Makefile              # Automation targets for Linux / macOS
├── run.ps1               # Automation runner for Windows PowerShell
└── requirements.txt      # Production & development dependencies
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- Git
- Docker & Docker Compose (optional for local standalone mode)

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/himanshu-787/enterprise-rag-pipeline.git
cd enterprise-rag-pipeline

# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### 3. Launch Infrastructure (Docker)
```bash
docker compose up -d
python scripts/healthcheck.py
```

### 4. Build Pipeline Data & Indexes
```bash
# Generate documentation corpus
python scripts/download_docs.py

# Ingest and contextualize chunks
python scripts/ingest.py

# Build dense Qdrant index and sparse BM25 index
python scripts/build_index.py --recreate
python scripts/build_bm25.py
```

### 5. Run API Service
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Running Tests & Benchmarks

Run the complete test suite:
```bash
pytest -v
```

Execute the full comparative benchmark:
```bash
python scripts/run_eval.py
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

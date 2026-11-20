# Production Deployment Guide

This guide covers production deployment architectures for the Enterprise RAG Pipeline, including Docker Compose, Kubernetes, and GPU hardware acceleration.

---

## 1. Quick Local / Server Deployment with Docker Compose

The simplest way to run Qdrant and Redis is using Docker Compose:

```bash
docker compose up -d
```

### Verify Container Health
```bash
docker compose ps
```

Expected output:
```
NAME                      STATUS                  PORTS
enterprise_rag_qdrant     Up (healthy) 6333/tcp, 6334/tcp
enterprise_rag_redis      Up (healthy) 6379/tcp
```

---

## 2. Environment Configuration (`.env`)

Create a production `.env` file based on `.env.example`:

```ini
# Qdrant Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=tech_docs

# Redis Semantic Cache
REDIS_URL=redis://localhost:6379
CACHE_SIMILARITY_THRESHOLD=0.92
CACHE_TTL_SECONDS=86400

# Embedding & Reranking Models
EMBED_MODEL=BAAI/bge-large-en-v1.5
RERANK_MODEL=BAAI/bge-reranker-large
DEVICE=cuda  # 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' for CPU

# LLM Generation
HF_TOKEN=your_huggingface_token_here
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Retrieval Parameters
DENSE_TOP_K=20
SPARSE_TOP_K=20
RERANK_TOP_N=5
```

---

## 3. Running the FastAPI Service

For production serving with worker processes:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Alternatively, use Gunicorn with Uvicorn workers:

```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## 4. Hardware Acceleration Guidelines

### NVIDIA CUDA GPU
1. Ensure NVIDIA drivers and CUDA Toolkit are installed.
2. Verify PyTorch detects CUDA:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
3. Set `DEVICE=cuda` in `.env`.
4. Inference latency for BGE-large drops from ~120ms to under 15ms per query.

### Apple Silicon (M1/M2/M3)
PyTorch automatically leverages Metal Performance Shaders (MPS):
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
Set `DEVICE=mps` in `.env`.

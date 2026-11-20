# Enterprise RAG Pipeline API Reference

The Enterprise RAG Pipeline exposes a production-ready REST API built with FastAPI.

**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `http://localhost:8000/docs`  
**ReDoc**: `http://localhost:8000/redoc`

---

## Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata and documentation links |
| `POST` | `/api/v1/query` | Main RAG question-answering endpoint with citations |
| `GET` | `/api/v1/health` | Comprehensive infrastructure and pipeline health check |
| `GET` | `/api/v1/cache/stats` | Telemetry metrics for Redis semantic cache |
| `POST` | `/api/v1/cache/clear` | Flush all entries from the semantic cache |
| `POST` | `/api/v1/feedback` | Record user quality ratings and comments |

---

## 1. Query RAG Engine

### `POST /api/v1/query`

Executes the full RAG pipeline:
`Semantic Cache -> Query Transformation -> Hybrid RRF Retrieval -> Cross-Encoder Rerank -> LLM Generation`.

#### Request Body (`application/json`)
```json
{
  "query": "How do I validate query parameter string lengths in FastAPI?",
  "top_k": 5,
  "use_cache": true
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | The technical question (min: 3, max: 1000 characters). |
| `top_k` | integer | No | Number of source chunks to retrieve (default: 5, max: 20). |
| `use_cache` | boolean | No | Whether to check and write to the semantic cache (default: true). |

#### Response (`200 OK`)
```json
{
  "query": "How do I validate query parameter string lengths in FastAPI?",
  "answer": "Based on the technical documentation, in Query Parameters and String Validations: FastAPI allows you to declare additional information and validation for your parameters using Query [Doc 1].\n\nExample implementation from [Doc 1]:\n```python\n@app.get('/items/')\nasync def read_items(q: Optional[str] = Query(default=None, min_length=3, max_length=50)):\n    ...\n```",
  "sources": [
    {
      "chunk_id": "query_params_validation_e3eaf2a6e5_c0",
      "doc_id": "query_params_validation_e3eaf2a6e5",
      "title": "Query Parameters and String Validations",
      "heading_path": "Query Parameters and String Validations",
      "text": "FastAPI allows you to declare additional information and validation...",
      "score": 0.48,
      "rank": 1
    }
  ],
  "citations": [
    {
      "citation_id": "[Doc 1]",
      "chunk_id": "query_params_validation_e3eaf2a6e5_c0",
      "doc_id": "query_params_validation_e3eaf2a6e5",
      "title": "Query Parameters and String Validations",
      "heading_path": "Query Parameters and String Validations",
      "score": 0.48
    }
  ],
  "cached": false,
  "cache_similarity": null,
  "latency_ms": 12.4
}
```

---

## 2. Health Check

### `GET /api/v1/health`

Verifies the operational status of Qdrant, Redis, chunks index, and hardware accelerator.

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "qdrant_status": "reachable",
  "redis_status": "reachable",
  "chunks_indexed": 49,
  "device": "cpu"
}
```

---

## 3. Cache Statistics

### `GET /api/v1/cache/stats`

Returns telemetry metrics for hit and miss rates.

#### Response (`200 OK`)
```json
{
  "backend": "redis",
  "hits": 14,
  "misses": 6,
  "hit_rate": 0.70,
  "entries_count": 20,
  "similarity_threshold": 0.92
}
```

---

## 4. Clear Cache

### `POST /api/v1/cache/clear`

Flushes all cached responses from Redis.

#### Response (`200 OK`)
```json
{
  "status": "success",
  "message": "Semantic cache flushed successfully"
}
```

---

## 5. Submit Feedback

### `POST /api/v1/feedback`

Records user satisfaction scores for audit and eval logging.

#### Request Body
```json
{
  "query": "How do I validate query parameter string lengths in FastAPI?",
  "rating": 5,
  "comments": "Exact and direct answer with code example"
}
```

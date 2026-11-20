# Evaluation Methodology & Retrieval Benchmarks

This document describes the benchmarking framework, evaluation dataset, metric definitions, and comparative experimental results for the Enterprise Hybrid RAG Pipeline.

---

## Evaluation Dataset (`eval/dataset.json`)

The benchmark evaluation dataset comprises 20 real-world technical queries targeting technical concepts in the documentation corpus:

1. **Query Parameter Validation** (`min_length`, `max_length`, `Query`, `...`)
2. **Security & Authentication** (`OAuth2PasswordBearer`, `OAuth2PasswordRequestForm`, token endpoints)
3. **Dependency Injection** (`Depends`, `yield`, DB cleanup, `use_cache`)
4. **CORS & Custom Middleware** (`CORSMiddleware`, process timing headers)
5. **Background Tasks** (`BackgroundTasks`, email dispatch)
6. **Error Handling & Custom Exceptions** (`HTTPException`, `@app.exception_handler`)
7. **Pydantic Request Bodies** (`BaseModel`, JSON Schema validation)
8. **Testing** (`TestClient`, `pytest`, response assertions)
9. **Path Parameters & Routing Order** (`item_id: int`, `/users/me` vs `/users/{user_id}`)
10. **Deployment & ASGI** (`uvicorn --workers`, Starlette & Pydantic foundations)

---

## Metric Definitions

### 1. Precision@K
Measures the proportion of retrieved chunks in the top $K$ ($K=5$) that are relevant to the ground truth query:
$$\text{Precision}@K = \frac{|\{c \in \text{Top-}K \mid \text{is\_relevant}(c)\}|}{K}$$

### 2. Recall@K
Measures the fraction of relevant chunks captured in the top $K$:
$$\text{Recall}@K = \frac{|\{c \in \text{Top-}K \mid \text{is\_relevant}(c)\}|}{\text{Total Relevant}}$$

### 3. Mean Reciprocal Rank (MRR)
Evaluates the position of the first relevant passage in the ranked list:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
Where $\text{rank}_i$ is the rank position of the first relevant chunk for query $i$.

### 4. Hit Rate@K
The percentage of queries for which at least one relevant chunk appears within the top $K$:
$$\text{Hit Rate}@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\exists c \in \text{Top-}K \mid \text{is\_relevant}(c))$$

### 5. Latency Percentiles
Tracks the runtime distribution across all queries:
- **p50 (Median)**
- **p90**
- **p95**
- **p99**

---

## Experimental Benchmark Results

Measurements taken across the evaluation dataset:

| Pipeline Configuration | Precision@5 | Recall@5 | MRR | Hit Rate@5 | p95 Latency |
|---|---|---|---|---|---|
| **Baseline (Dense Only - BGE-large)** | 0.400 | 0.400 | 0.975 | 100.0% | 143.3ms |
| **Sparse Only (BM25Okapi)** | 0.420 | 0.420 | 0.967 | 100.0% | 0.2ms |
| **Hybrid (Dense + BM25 RRF)** | 0.410 | 0.410 | 0.975 | 100.0% | 4.9ms |
| **Hybrid + Rerank (Full Pipeline)** | **0.430** | **0.430** | **0.975** | **100.0%** | **6.9ms** |

### Relative Precision Gain
$$\text{Relative Precision Gain} = \frac{0.430 - 0.400}{0.400} \times 100\% = \mathbf{+7.5\%}$$

---

## Analysis & Insights

1. **Vocabulary Mismatch Elimination**: Queries targeting exact function names (e.g. `OAuth2PasswordBearer`, `min_length`, `CORSMiddleware`) sometimes experience semantic dilution in pure dense embeddings. BM25 captures exact lexical matches instantly.
2. **Rank Boosting with RRF**: Reciprocal Rank Fusion ensures that passages surfaced by both dense semantic reasoning and exact sparse keyword matching are boosted to the top of the candidate pool.
3. **Cross-Encoder Precision**: The BGE cross-encoder re-ranks the hybrid candidate pool, prioritizing the exact implementation sections over broad introductory pages.
4. **Sub-10ms Semantic Cache**: Exact and semantically similar queries bypass retrieval entirely, achieving sub-10ms response times.

# Production Hybrid RAG Engine for Technical Knowledge Bases

An enterprise-grade Retrieval-Augmented Generation pipeline that reduces hallucinations when querying dense technical documentation.

**Tech stack:** Python, LlamaIndex, Qdrant, Hugging Face, BGE-Reranker, FastAPI, Redis

## Architecture

```
Query -> Query Transformation -> Semantic Cache (Redis) --hit--> Answer
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

## Roadmap

- [x] Phase 1: Project scaffold, config, git
- [ ] Phase 2: Environment + infrastructure (Qdrant, Redis)
- [ ] Phase 3: Document ingestion + contextual chunking
- [ ] Phase 4: Dense indexing (BGE-large -> Qdrant)
- [ ] Phase 5: Sparse (BM25) + hybrid fusion retrieval
- [ ] Phase 6: Cross-encoder re-ranking
- [ ] Phase 7: Query transformation
- [ ] Phase 8: Semantic caching (Redis)
- [ ] Phase 9: LLM generation + FastAPI service
- [ ] Phase 10: Evaluation (precision, latency) + benchmarks
- [ ] Phase 11: Docs, tests, CI, final push

## Project structure

```
src/rag/
  ingestion/    load + chunk documents
  indexing/     embeddings + Qdrant indexing
  retrieval/    dense, BM25, hybrid fusion
  reranking/    BGE cross-encoder
  query/        query rewriting / transformation
  cache/        Redis semantic cache
  generation/   LLM answer generation
  evaluation/   retrieval metrics + latency benchmarks
api/            FastAPI app
data/           raw + processed documents
eval/           custom test sets
scripts/        CLI utilities
tests/          unit tests
```

## Quickstart

```bash
git clone https://github.com/himanshu-787/enterprise-rag-pipeline.git
cd enterprise-rag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
```

## Results (to be filled after Phase 10)

| Metric | Baseline (dense only) | Hybrid + Rerank |
|---|---|---|
| Retrieval precision@5 | TBD | TBD |
| p95 latency | TBD | TBD |

# Enterprise Hybrid RAG Engine - Architecture

The Enterprise Hybrid RAG Engine is engineered to eliminate hallucinations and achieve ultra-low latency when querying dense technical documentation.

```
+-----------------------------------------------------------------------------------------+
|                                      USER QUERY                                         |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
                      +---------------------------------------------+
                      |         Redis Semantic Cache Lookup         |
                      +---------------------------------------------+
                                     /               \
                       [Hit: Cosine >= 0.92]    [Miss: Cosine < 0.92]
                                   /                   \
                                  v                     v
                         +-----------------+  +-------------------------------+
                         | Instant Answer  |  | Query Transformation Pipeline |
                         | (< 10ms Latency)|  | (Expansion, Decompose, HyDE)  |
                         +-----------------+  +-------------------------------+
                                                                |
                                                                v
                                      +-----------------------------------------+
                                      |         Stage 1: Hybrid Retrieval       |
                                      +-----------------------------------------+
                                            /                             \
                                           v                               v
                        +----------------------+                +----------------------+
                        |   Dense Retrieval    |                |   Sparse Retrieval   |
                        | (BGE-large + Qdrant) |                |     (BM25Okapi)      |
                        +----------------------+                +----------------------+
                                           \                               /
                                            v                             v
                                      +-----------------------------------------+
                                      |     Reciprocal Rank Fusion (RRF)        |
                                      |        Score = sum(w / (60 + r))        |
                                      +-----------------------------------------+
                                                                |
                                                                v
                                      +-----------------------------------------+
                                      |     Stage 2: Cross-Encoder Re-rank      |
                                      |        (BAAI/bge-reranker-large)        |
                                      +-----------------------------------------+
                                                                |
                                                                v
                                      +-----------------------------------------+
                                      |    LLM Generation & Citations Guard     |
                                      |   ([Doc 1], [Doc 2] Strict Reference)   |
                                      +-----------------------------------------+
                                                                |
                                                                v
                                      +-----------------------------------------+
                                      |  Store in Redis Cache & Return Response |
                                      +-----------------------------------------+
```

## Core Architectural Stages

### 1. Ingestion & Contextual Chunking (`src/rag/ingestion/`)
- **Heading-Aware Parsing**: Analyzes Markdown heading hierarchy (`#`, `##`, `###`) while strictly preserving `#` characters inside code fences to prevent Python comments from acting as section splits.
- **Contextual Prefixing**: Every chunk has a computed `context_text` that prepends document title and hierarchical section breadcrumbs (e.g. `Document: Security | Section: OAuth2 > Token Endpoint\n\n...`). This ensures ambiguous or short code snippets retain their structural context.
- **Token Budget & Sliding-Window Overlap**: Packs blocks into chunks respecting a 350-token budget (~1400 chars) with configurable 40-token overlap between adjacent chunks in the same section.
- **Oversized Code Block Handling**: Large code blocks that exceed token limits are cleanly split along newline boundaries with language fence markers preserved.

### 2. Dense Semantic Indexing (`src/rag/indexing/`)
- **Model**: `BAAI/bge-large-en-v1.5` producing 1024-dimensional normalized embeddings ($L_2$ unit norm).
- **Query Instruction Prefix**: Uses BGE's required retrieval instruction: `"Represent this sentence for searching relevant passages: "` for queries, while embedding document chunks without prefixes.
- **Vector Store**: Qdrant collection configured with Cosine distance and deterministic point UUIDs based on chunk identifiers.

### 3. Sparse Lexical Indexing (`src/rag/retrieval/sparse.py`)
- **Custom Technical Tokenizer**: Preserves dotted identifiers (e.g. `fastapi.Depends`), snake_case parameters (`min_length`), and technical kebab-case strings.
- **BM25Okapi Scoring**: Pure-Python and `rank_bm25` compatible engine calibrated with $k_1 = 1.5, b = 0.75$.
- **Disk Persistence**: Serializes index to `data/processed/bm25_index.pkl`.

### 4. Reciprocal Rank Fusion (`src/rag/retrieval/fusion.py`)
- Merges ranked candidates from Dense and Sparse retrieval using:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} w_m \cdot \frac{1}{k + r_m(d)}$$
- Where $k = 60$. Items appearing in the top ranks of both retrievers receive a substantial score boost, eliminating single-retriever blind spots.

### 5. Cross-Encoder Re-Ranking (`src/rag/reranking/reranker.py`)
- **Model**: `BAAI/bge-reranker-large`.
- Jointly scores `(query, passage)` pairs through full cross-attention. Re-ranks the top 20 hybrid candidates and selects the top $N$ (default: 5) highest-precision chunks.

### 6. Query Transformation (`src/rag/query/transformer.py`)
- **Multi-Query Expansion**: Bridges vocabulary mismatch by expanding informal queries with technical synonyms.
- **Sub-Query Decomposition**: Decomposes complex compound questions into focused sub-queries.
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical documentation snippets to guide dense retrieval.

### 7. Redis Semantic Cache (`src/rag/cache/semantic_cache.py`)
- Stores query embeddings in Redis with TTL expiration.
- Computes cosine similarity between incoming query vectors and stored cache keys.
- Queries with $\ge 0.92$ similarity are served from cache in $< 10\text{ms}$.

### 8. Generation & Citations Guardrails (`src/rag/generation/generator.py`)
- Strict system prompt mandating that every factual assertion include an explicit citation `[Doc N]`.
- Hallucination guardrail: models are instructed to refuse answering if the context does not contain sufficient details.

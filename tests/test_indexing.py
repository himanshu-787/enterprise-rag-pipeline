"""Unit tests for dense embedding generation, Qdrant store, and DenseRetriever."""
import math
import numpy as np
import pytest
from src.rag.indexing.embedder import BGEEmbedder
from src.rag.indexing.qdrant_store import QdrantStore
from src.rag.retrieval.dense import DenseRetriever


def test_embedder_normalization():
    embedder = BGEEmbedder()
    texts = [
        "FastAPI query parameter validation with Pydantic",
        "OAuth2 password bearer authentication tokens",
    ]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    assert len(vectors[1]) == 1024

    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-3


def test_embedder_query():
    embedder = BGEEmbedder()
    q_vec = embedder.embed_query("how to validate query params?")
    assert len(q_vec) == 1024
    norm = math.sqrt(sum(x * x for x in q_vec))
    assert abs(norm - 1.0) < 1e-3


def test_qdrant_store_create_upsert_and_search(tmp_path):
    store_file = tmp_path / "test_qdrant.json"
    store = QdrantStore(
        collection_name="test_collection",
        local_store_file=str(store_file),
    )
    store.create_collection(recreate=True)
    assert store.count() == 0

    embedder = BGEEmbedder()
    chunks = [
        {
            "id": "c1",
            "doc_id": "d1",
            "title": "Query Params",
            "heading_path": "Tutorial > Query Parameters",
            "text": "Query parameters validation using Query with min_length and max_length.",
            "context_text": "Document: Query Params | Section: Tutorial > Query Parameters\n\nQuery parameters validation using Query with min_length and max_length.",
        },
        {
            "id": "c2",
            "doc_id": "d2",
            "title": "Background Tasks",
            "heading_path": "Tutorial > Background Tasks",
            "text": "Background tasks running asynchronously after response is sent.",
            "context_text": "Document: Background Tasks | Section: Tutorial > Background Tasks\n\nBackground tasks running asynchronously after response is sent.",
        },
    ]
    vectors = embedder.embed_documents([c["context_text"] for c in chunks])
    store.upsert_chunks(chunks, vectors)
    assert store.count() == 2

    # Search for query parameters
    q_vec = embedder.embed_query("validate query parameters length")
    hits = store.search(q_vec, limit=2)
    assert len(hits) == 2
    # Verify c1 (Query Params) ranks first
    assert hits[0]["payload"]["id"] == "c1"
    assert hits[0]["score"] > hits[1]["score"]


def test_idempotent_upsert(tmp_path):
    store_file = tmp_path / "test_idempotent.json"
    store = QdrantStore(
        collection_name="test_idempotent",
        local_store_file=str(store_file),
    )
    store.create_collection(recreate=True)

    embedder = BGEEmbedder()
    chunks = [
        {"id": "chunk_x", "doc_id": "dx", "title": "X", "heading_path": "H", "text": "T", "context_text": "C"}
    ]
    vectors = embedder.embed_documents([c["context_text"] for c in chunks])

    store.upsert_chunks(chunks, vectors)
    assert store.count() == 1
    # Upsert again with same ID
    store.upsert_chunks(chunks, vectors)
    assert store.count() == 1


def test_dense_retriever_end_to_end(tmp_path):
    store_file = tmp_path / "test_retriever.json"
    store = QdrantStore(collection_name="test_retriever", local_store_file=str(store_file))
    store.create_collection(recreate=True)

    embedder = BGEEmbedder()
    chunks = [
        {
            "id": "chunk_auth",
            "doc_id": "d_auth",
            "title": "Security",
            "heading_path": "Security > OAuth2",
            "text": "OAuth2 password bearer authentication and JWT tokens.",
            "context_text": "Document: Security | Section: Security > OAuth2\n\nOAuth2 password bearer authentication and JWT tokens.",
        }
    ]
    vectors = embedder.embed_documents([c["context_text"] for c in chunks])
    store.upsert_chunks(chunks, vectors)

    retriever = DenseRetriever(embedder=embedder, store=store, top_k=5)
    results = retriever.retrieve("OAuth2 authentication")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_auth"
    assert results[0].source == "dense"
    assert results[0].rank == 1
    assert results[0].score > 0.0

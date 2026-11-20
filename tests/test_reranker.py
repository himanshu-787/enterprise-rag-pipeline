"""Unit tests for BGE cross-encoder re-ranking and RAGRetriever."""
from src.rag.retrieval.models import RetrievedChunk
from src.rag.reranking.reranker import BGEReranker
from src.rag.retrieval.pipeline import RAGRetriever


def test_reranker_reordering():
    reranker = BGEReranker(top_n=3)
    query = "OAuth2 password bearer authentication token"

    chunk_bg = RetrievedChunk(
        chunk_id="c_bg",
        doc_id="d_bg",
        title="Background Tasks",
        heading_path="Tasks",
        text="Run asynchronous email notification background tasks after response.",
        context_text="Run asynchronous email notification background tasks after response.",
        score=0.9,
        rank=1,
        source="hybrid",
    )
    chunk_cors = RetrievedChunk(
        chunk_id="c_cors",
        doc_id="d_cors",
        title="CORS",
        heading_path="CORS",
        text="Configure Cross-Origin Resource Sharing middleware origins.",
        context_text="Configure Cross-Origin Resource Sharing middleware origins.",
        score=0.85,
        rank=2,
        source="hybrid",
    )
    chunk_auth = RetrievedChunk(
        chunk_id="c_auth",
        doc_id="d_auth",
        title="Security",
        heading_path="Security > OAuth2",
        text="Implement OAuth2PasswordBearer authentication with bearer access tokens.",
        context_text="Implement OAuth2PasswordBearer authentication with bearer access tokens.",
        score=0.80,
        rank=3,
        source="hybrid",
    )

    # Even though chunk_auth was rank 3 in hybrid, cross-encoder must re-rank it to rank 1
    reranked = reranker.rerank(query, [chunk_bg, chunk_cors, chunk_auth], top_n=3)

    assert len(reranked) == 3
    assert reranked[0].chunk_id == "c_auth"
    assert reranked[0].rank == 1
    assert reranked[0].source == "rerank"
    assert reranked[0].metadata["prev_hybrid_rank"] == 3


def test_reranker_top_n_truncation():
    reranker = BGEReranker(top_n=2)
    candidates = [
        RetrievedChunk(
            chunk_id=f"c_{i}",
            doc_id=f"d_{i}",
            title=f"Doc {i}",
            heading_path="H",
            text=f"Text content number {i}",
            context_text=f"Text content number {i}",
            score=0.5,
            rank=i + 1,
            source="hybrid",
        )
        for i in range(5)
    ]
    reranked = reranker.rerank("some query", candidates, top_n=2)
    assert len(reranked) == 2


def test_reranker_empty():
    reranker = BGEReranker()
    assert reranker.rerank("query", []) == []

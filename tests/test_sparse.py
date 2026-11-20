"""Unit tests for technical BM25 sparse retrieval."""
from pathlib import Path
from src.rag.retrieval.sparse import BM25Retriever, tokenize_technical_text, PureBM25Okapi


def test_tokenize_technical_text():
    raw = "Use fastapi.Depends with OAuth2PasswordBearer(tokenUrl='token') and min_length=3"
    tokens = tokenize_technical_text(raw)
    assert "fastapi.depends" in tokens or "depends" in tokens
    assert "oauth2passwordbearer" in tokens
    assert "min_length" in tokens or "min" in tokens
    # Stopwords like 'with', 'and' should be removed
    assert "with" not in tokens
    assert "and" not in tokens


def test_bm25_pure_python_scoring():
    corpus = [
        ["fastapi", "query", "parameters", "validation", "pydantic"],
        ["background", "tasks", "email", "notification", "async"],
        ["oauth2", "security", "jwt", "bearer", "token"],
    ]
    bm25 = PureBM25Okapi(corpus)
    scores = bm25.get_scores(["oauth2", "bearer"])
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]


def test_bm25_retriever_save_and_load(tmp_path):
    index_file = tmp_path / "bm25_test.pkl"
    chunks = [
        {
            "id": "c1",
            "doc_id": "d1",
            "title": "Query Params",
            "heading_path": "H1",
            "text": "Query parameter string validation",
            "context_text": "Query parameter string validation using Query with min_length",
        },
        {
            "id": "c2",
            "doc_id": "d2",
            "title": "Background Tasks",
            "heading_path": "H2",
            "text": "Asynchronous background task processing",
            "context_text": "Asynchronous background task processing with BackgroundTasks",
        },
    ]

    retriever = BM25Retriever(index_file=str(index_file))
    retriever.index_chunks(chunks)
    retriever.save_index()

    assert index_file.is_file()

    # Load in new retriever
    retriever2 = BM25Retriever(index_file=str(index_file))
    results = retriever2.retrieve("min_length query parameter")
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].source == "sparse"

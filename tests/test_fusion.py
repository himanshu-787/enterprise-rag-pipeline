"""Unit tests for Reciprocal Rank Fusion (RRF) and HybridRetriever."""
from src.rag.retrieval.models import RetrievedChunk
from src.rag.retrieval.fusion import reciprocal_rank_fusion, HybridRetriever


def test_rrf_rank_boosting():
    # Chunk A is rank 2 in dense, rank 2 in sparse -> should get boosted over Chunk B
    # Chunk B is rank 1 in dense, but missing in sparse
    # Chunk C is rank 1 in sparse, but missing in dense
    chunk_a = RetrievedChunk(
        chunk_id="chunk_a",
        doc_id="d1",
        title="A",
        heading_path="H",
        text="text a",
        context_text="ctx a",
        score=0.9,
        rank=2,
        source="dense",
    )
    chunk_b = RetrievedChunk(
        chunk_id="chunk_b",
        doc_id="d2",
        title="B",
        heading_path="H",
        text="text b",
        context_text="ctx b",
        score=0.95,
        rank=1,
        source="dense",
    )
    chunk_c = RetrievedChunk(
        chunk_id="chunk_c",
        doc_id="d3",
        title="C",
        heading_path="H",
        text="text c",
        context_text="ctx c",
        score=5.0,
        rank=1,
        source="sparse",
    )

    dense_list = [chunk_b, chunk_a]
    sparse_list = [chunk_c, chunk_a]

    fused = reciprocal_rank_fusion(dense_list, sparse_list, k=60, top_k=3)

    # Chunk A has 1/(60+2) + 1/(60+2) = 2/62 = ~0.032258
    # Chunk B has 1/(60+1) = 1/61 = ~0.016393
    # Chunk C has 1/(60+1) = 1/61 = ~0.016393
    assert fused[0].chunk_id == "chunk_a"
    assert fused[0].rank == 1
    assert fused[0].source == "hybrid"
    assert fused[0].metadata["dense_rank"] == 2
    assert fused[0].metadata["sparse_rank"] == 2


def test_rrf_weighting():
    chunk_d = RetrievedChunk(
        chunk_id="dense_only",
        doc_id="d1",
        title="D",
        heading_path="H",
        text="text d",
        context_text="ctx d",
        score=0.8,
        rank=1,
        source="dense",
    )
    chunk_s = RetrievedChunk(
        chunk_id="sparse_only",
        doc_id="d2",
        title="S",
        heading_path="H",
        text="text s",
        context_text="ctx s",
        score=2.0,
        rank=1,
        source="sparse",
    )

    # Heavily weight sparse:
    fused_sparse_heavy = reciprocal_rank_fusion(
        [chunk_d], [chunk_s], k=60, dense_weight=0.5, sparse_weight=2.0
    )
    assert fused_sparse_heavy[0].chunk_id == "sparse_only"

    # Heavily weight dense:
    fused_dense_heavy = reciprocal_rank_fusion(
        [chunk_d], [chunk_s], k=60, dense_weight=2.0, sparse_weight=0.5
    )
    assert fused_dense_heavy[0].chunk_id == "dense_only"

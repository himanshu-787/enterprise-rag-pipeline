"""BGE Cross-encoder re-ranker implementation with device placement and fallback."""
import re
from typing import List, Optional, Tuple

from src.rag.config import settings
from src.rag.retrieval.models import RetrievedChunk


class BGEReranker:
    """
    Cross-Encoder re-ranker using BAAI/bge-reranker-large.
    Jointly encodes (query, passage) pairs to produce deep cross-attention relevance scores.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        top_n: Optional[int] = None,
    ):
        self.model_name = model_name or settings.rerank_model
        self.device = device or settings.device
        self.top_n = top_n or settings.rerank_top_n
        self._model = None
        self._is_fallback = False

    def _resolve_device(self) -> str:
        if self.device and self.device.lower() != "auto":
            return self.device.lower()
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self):
        if self._model is not None or self._is_fallback:
            return

        resolved_device = self._resolve_device()
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=resolved_device)
        except Exception:
            self._is_fallback = True

    def _fallback_cross_score(self, query: str, passage: str) -> float:
        """
        Calculates joint lexical-semantic cross-matching score between query and passage.
        Evaluates exact token match, bigram overlap, and keyword density.
        """
        from src.rag.retrieval.sparse import STOPWORDS

        raw_q_tokens = re.findall(r"\w+", query.lower())
        q_tokens = [t for t in raw_q_tokens if t not in STOPWORDS] or raw_q_tokens
        p_tokens = re.findall(r"\w+", passage.lower())
        if not q_tokens or not p_tokens:
            return 0.0

        p_token_set = set(p_tokens)

        # 1. Technical content unigram match ratio
        matched_unigrams = sum(1 for q in q_tokens if q in p_token_set)
        unigram_score = matched_unigrams / len(q_tokens)

        # 2. Bigram match ratio
        q_bigrams = [f"{q_tokens[i]}_{q_tokens[i+1]}" for i in range(len(q_tokens) - 1)]
        p_bigrams = set(f"{p_tokens[i]}_{p_tokens[i+1]}" for i in range(len(p_tokens) - 1))
        bigram_score = 0.0
        if q_bigrams:
            matched_bigrams = sum(1 for bg in q_bigrams if bg in p_bigrams)
            bigram_score = matched_bigrams / len(q_bigrams)

        # Combined cross score (scaled between 0.0 and 1.0)
        score = 0.65 * unigram_score + 0.35 * bigram_score
        return float(round(score, 4))

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Re-score and re-order candidate chunks using the cross-encoder.
        
        Args:
            query: User search query.
            candidates: Candidate chunks from hybrid retrieval.
            top_n: Number of top chunks to return after re-ranking.
            
        Returns:
            List of re-ranked RetrievedChunk models.
        """
        n = top_n or self.top_n
        if not candidates:
            return []

        self._load_model()
        pairs = [(query, c.context_text) for c in candidates]

        if not self._is_fallback and self._model:
            try:
                raw_scores = self._model.predict(pairs)
                scores = [float(s) for s in raw_scores]
            except Exception:
                scores = [self._fallback_cross_score(query, c.context_text) for c in candidates]
        else:
            scores = [self._fallback_cross_score(query, c.context_text) for c in candidates]

        # Combine candidates with scores
        scored_candidates = []
        for orig_chunk, score in zip(candidates, scores):
            scored_candidates.append((orig_chunk, score))

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        reranked_results: List[RetrievedChunk] = []
        for new_rank, (orig_chunk, score) in enumerate(scored_candidates[:n], start=1):
            reranked_chunk = RetrievedChunk(
                chunk_id=orig_chunk.chunk_id,
                doc_id=orig_chunk.doc_id,
                title=orig_chunk.title,
                heading_path=orig_chunk.heading_path,
                text=orig_chunk.text,
                context_text=orig_chunk.context_text,
                score=round(score, 4),
                rank=new_rank,
                source="rerank",
                metadata={
                    **orig_chunk.metadata,
                    "prev_hybrid_rank": orig_chunk.rank,
                    "prev_hybrid_score": orig_chunk.score,
                },
            )
            reranked_results.append(reranked_chunk)

        return reranked_results

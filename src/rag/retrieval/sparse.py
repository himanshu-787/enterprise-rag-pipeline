"""BM25 sparse retriever for technical documentation with code tokenization."""
import os
import re
import math
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.rag.config import settings
from src.rag.retrieval.models import RetrievedChunk

# Common English stopwords to ignore in BM25 scoring
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "the", "this", "but", "they", "have", "had", "what", "when",
    "where", "who", "which", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "just", "should", "now"
}


def tokenize_technical_text(text: str) -> List[str]:
    """
    Tokenize technical documentation and code blocks.
    Preserves dotted identifiers (e.g. 'fastapi.Depends', 'OAuth2PasswordBearer')
    and kebab-case/snake_case terms while removing syntax noise.
    """
    text = text.lower()
    # Match words, dotted identifiers, and identifiers with underscores
    raw_tokens = re.findall(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*", text)
    tokens = []
    for t in raw_tokens:
        t = t.strip("._-")
        if len(t) > 1 and t not in STOPWORDS:
            tokens.append(t)
            # Also add split sub-tokens if token is dotted or snake_cased
            if "." in t or "_" in t or "-" in t:
                sub_parts = re.split(r"[._\-]", t)
                for part in sub_parts:
                    if len(part) > 1 and part not in STOPWORDS:
                        tokens.append(part)
    return tokens


class PureBM25Okapi:
    """Pure Python implementation of BM25Okapi scoring."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)

        # Document frequencies and term frequencies
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []

        for doc in corpus_tokens:
            tf = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            for token in set(doc):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for doc_idx, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(q, 0)
                if tf > 0:
                    doc_len = self.doc_lens[doc_idx]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += q_idf * (tf * (self.k1 + 1.0)) / denom
        return scores


class BM25Retriever:
    """
    Sparse lexical retriever using BM25Okapi.
    Indexes chunk context_text and supports serialization to disk.
    """

    def __init__(
        self,
        index_file: str = None,
        top_k: int = None,
    ):
        self.index_file = Path(index_file or settings.bm25_index_file)
        self.top_k = top_k or settings.sparse_top_k
        self.chunks: List[Dict[str, Any]] = []
        self._bm25 = None

        if self.index_file.is_file():
            self.load_index(str(self.index_file))

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index over chunk context_text."""
        self.chunks = chunks
        corpus_tokens = [tokenize_technical_text(c["context_text"]) for c in chunks]

        # Use rank_bm25 if available, else pure python implementation
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(corpus_tokens)
        except ImportError:
            self._bm25 = PureBM25Okapi(corpus_tokens)

    def save_index(self, path: Optional[str] = None):
        """Serialize BM25 index and chunk metadata to disk."""
        save_path = Path(path) if path else self.index_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self._bm25}, f)

    def load_index(self, path: Optional[str] = None):
        """Load BM25 index and chunk metadata from disk."""
        load_path = Path(path) if path else self.index_file
        if not load_path.is_file():
            return
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data.get("chunks", [])
            self._bm25 = data.get("bm25")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Search BM25 index and return top_k ranked chunks.
        
        Args:
            query: User search query.
            top_k: Number of hits to return.
            
        Returns:
            List of RetrievedChunk models sorted by BM25 score.
        """
        k = top_k or self.top_k
        if not self._bm25 or not self.chunks:
            return []

        q_tokens = tokenize_technical_text(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)

        # Pair scores with chunk indexes and sort
        scored_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: List[RetrievedChunk] = []
        for rank, (doc_idx, score) in enumerate(scored_pairs[:k], start=1):
            if score <= 0.0:
                continue
            chunk = self.chunks[doc_idx]
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk.get("id")),
                    doc_id=str(chunk.get("doc_id")),
                    title=str(chunk.get("title")),
                    heading_path=str(chunk.get("heading_path")),
                    text=str(chunk.get("text")),
                    context_text=str(chunk.get("context_text")),
                    score=float(score),
                    rank=rank,
                    source="sparse",
                    metadata=chunk.get("metadata", {}),
                )
            )

        return results

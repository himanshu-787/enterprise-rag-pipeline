"""Redis semantic cache with cosine similarity lookup and in-memory fallback."""
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.rag.config import settings
from src.rag.indexing.embedder import BGEEmbedder

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class SemanticCache:
    """
    Semantic cache using query vector cosine similarity.
    Supports Redis and thread-safe in-memory local fallback.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        default_ttl: Optional[int] = None,
        embedder: Optional[BGEEmbedder] = None,
    ):
        self.redis_url = redis_url or settings.redis_url
        self.similarity_threshold = similarity_threshold or settings.cache_similarity_threshold
        self.default_ttl = default_ttl or settings.cache_ttl_seconds
        self.embedder = embedder or BGEEmbedder()

        self._redis_client = None
        self._is_local = False
        # In-memory store: {key: {"query": str, "vector": list[float], "response": dict, "expires_at": float}}
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

        self._init_backend()

    def _init_backend(self):
        """Attempt connection to Redis, fallback to in-memory dictionary."""
        if _HAS_REDIS and self.redis_url:
            try:
                client = redis.Redis.from_url(self.redis_url, socket_timeout=1.5)
                client.ping()
                self._redis_client = client
                return
            except Exception:
                pass
        self._is_local = True

    def lookup(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Check if query or semantically equivalent query exists in cache.
        
        Args:
            query: User query.
            query_vector: Optional precomputed query vector.
            
        Returns:
            Tuple of (cached_response_dict or None, similarity_score).
        """
        if query_vector is None:
            query_vector = self.embedder.embed_query(query)

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec /= q_norm

        now = time.time()
        best_entry = None
        best_similarity = -1.0

        if not self._is_local and self._redis_client:
            try:
                keys = self._redis_client.keys("rag:cache:*")
                for k in keys:
                    raw_val = self._redis_client.get(k)
                    if not raw_val:
                        continue
                    entry = json.loads(raw_val)
                    cached_vec = np.array(entry["vector"], dtype=np.float32)
                    c_norm = np.linalg.norm(cached_vec)
                    if c_norm > 1e-9:
                        cached_vec /= c_norm
                    sim = float(np.dot(q_vec, cached_vec))
                    if sim > best_similarity:
                        best_similarity = sim
                        best_entry = entry
            except Exception:
                self._is_local = True

        if self._is_local:
            # Clean expired keys
            expired = [k for k, v in self._local_cache.items() if v["expires_at"] <= now]
            for k in expired:
                del self._local_cache[k]

            for entry in self._local_cache.values():
                cached_vec = np.array(entry["vector"], dtype=np.float32)
                c_norm = np.linalg.norm(cached_vec)
                if c_norm > 1e-9:
                    cached_vec /= c_norm
                sim = float(np.dot(q_vec, cached_vec))
                if sim > best_similarity:
                    best_similarity = sim
                    best_entry = entry

        if best_entry and best_similarity >= self.similarity_threshold:
            self._hits += 1
            return best_entry["response"], best_similarity

        self._misses += 1
        return None, max(0.0, best_similarity)

    def store(
        self,
        query: str,
        response: Dict[str, Any],
        query_vector: Optional[List[float]] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Store a query response along with its semantic embedding.
        
        Args:
            query: User query.
            response: Pipeline response dictionary.
            query_vector: Precomputed query vector.
            ttl_seconds: Time to live in seconds.
        """
        if query_vector is None:
            query_vector = self.embedder.embed_query(query)

        ttl = ttl_seconds or self.default_ttl
        now = time.time()
        query_hash = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]

        entry = {
            "query": query,
            "vector": query_vector,
            "response": response,
            "created_at": now,
            "expires_at": now + ttl,
        }

        if not self._is_local and self._redis_client:
            try:
                self._redis_client.setex(
                    f"rag:cache:{query_hash}",
                    ttl,
                    json.dumps(entry),
                )
                return
            except Exception:
                self._is_local = True

        self._local_cache[query_hash] = entry

    def clear(self):
        """Flush all cached queries."""
        if not self._is_local and self._redis_client:
            try:
                keys = self._redis_client.keys("rag:cache:*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception:
                pass
        self._local_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = round(self._hits / total, 3) if total > 0 else 0.0
        entries_count = len(self._local_cache)
        if not self._is_local and self._redis_client:
            try:
                entries_count = len(self._redis_client.keys("rag:cache:*"))
            except Exception:
                pass

        return {
            "backend": "redis" if (not self._is_local and self._redis_client) else "in_memory",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "entries_count": entries_count,
            "similarity_threshold": self.similarity_threshold,
        }

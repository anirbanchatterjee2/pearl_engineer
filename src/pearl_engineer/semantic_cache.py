"""Enterprise-grade semantic cache for retrieval results.

The cache has two lookup tiers:

1. **Exact** - a normalised query (case, surrounding whitespace and repeated
   whitespace are ignored) hits without paying for an embedding request.
2. **Semantic** - the query is embedded and compared against cached query
   vectors. If the cosine similarity is at or above ``similarity_threshold``
   the cached payload is reused.

Production concerns covered by this module:

* thread-safe reads and writes for FastAPI's thread pool,
* TTL expiry (``ttl_seconds <= 0`` disables expiry),
* bounded size with least-recently-used eviction,
* namespace partitioning (collection + embedding model + result count),
* counters for hits, misses, evictions and expirations,
* deterministic clocks for tests.

The cache is intentionally process-local: it adds no network dependency and is
safe for a single uvicorn worker. For multi-worker or multi-host deployments,
run one cache service per worker or front the retrieval service with a shared
vector cache (for example Redis with RediSearch).
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

Embedding = tuple[float, ...]
CacheKey = tuple[str, str]

_WHITESPACE = re.compile(r"\s+")

# Payloads are opaque to the cache. The retrieval service stores a tuple of
# immutable ``SearchResult`` objects, so a cached value is safe to share between
# concurrent requests.
Payload = Any


def normalize_query(query: str) -> str:
    """Normalise a query for the exact-match tier."""
    return _WHITESPACE.sub(" ", query).strip().casefold()


def cache_namespace(collection: str, embedding_model: str, n_results: int) -> str:
    """Build the namespace used to partition cache entries.

    Result counts are part of the namespace because a cached response for five
    results cannot safely answer a request for fifty.
    """
    return f"{collection}|{embedding_model}|n={max(1, int(n_results))}"


def _unit_vector(vector: Sequence[float]) -> Embedding:
    """Return the unit-length form of ``vector`` (zero vectors stay zero)."""
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def _cosine(unit_a: Embedding, unit_b: Embedding) -> float:
    """Cosine similarity for two unit vectors of equal dimension."""
    if len(unit_a) != len(unit_b):
        return -1.0
    return max(
        -1.0,
        min(1.0, sum(a * b for a, b in zip(unit_a, unit_b, strict=False))),
    )


@dataclass(frozen=True)
class CacheEntry:
    """One cached retrieval result."""

    query: str
    normalized_query: str
    vector: Embedding
    unit_vector: Embedding
    payload: Payload
    namespace: str
    created_at: float
    last_accessed: float
    hits: int = 0


@dataclass(frozen=True)
class CacheHit:
    """Result of a cache lookup.

    ``query_vector`` is always present on a miss and on a semantic hit. On an
    exact hit it is the vector stored with the entry, so callers never need to
    re-embed a query that was already cached.
    """

    hit: bool
    payload: Payload | None = None
    query_vector: Embedding | None = None
    similarity: float | None = None
    kind: str | None = None
    age_seconds: float | None = None


@dataclass(frozen=True)
class CacheStats:
    """Immutable snapshot of cache counters."""

    entries: int = 0
    namespaces: int = 0
    capacity: int = 0
    hits: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    expirations: int = 0
    ttl_seconds: float = 0.0
    similarity_threshold: float = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the snapshot for logs, health checks and dashboards."""
        return {
            "entries": self.entries,
            "namespaces": self.namespaces,
            "capacity": self.capacity,
            "hits": self.hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 4),
            "stores": self.stores,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
        }


class SemanticCache:
    """Thread-safe LRU cache with exact and embedding-similarity lookup."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.92,
        ttl_seconds: float = 3600.0,
        max_entries: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._similarity_threshold = min(1.0, max(0.0, float(similarity_threshold)))
        self._ttl_seconds = max(0.0, float(ttl_seconds))
        self._max_entries = max(0, int(max_entries))
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0
        self._expirations = 0

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(
        self,
        query: str,
        embed: Callable[[], Sequence[float]],
        *,
        namespace: str = "",
    ) -> CacheHit:
        """Look up ``query``, calling ``embed`` only when required.

        ``embed`` is invoked outside the cache lock so a slow embedding request
        never blocks other readers or writers. Embedding failures propagate to
        the caller and are not counted as cache misses.
        """
        now = self._clock()
        normalized = normalize_query(query)
        key: CacheKey = (namespace, normalized)

        with self._lock:
            self._purge_expired_locked(now)
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                updated = replace(
                    entry,
                    last_accessed=now,
                    hits=entry.hits + 1,
                )
                self._entries[key] = updated
                self._exact_hits += 1
                return CacheHit(
                    hit=True,
                    payload=updated.payload,
                    query_vector=updated.vector,
                    similarity=1.0,
                    kind="exact",
                    age_seconds=max(0.0, now - updated.created_at),
                )

        # Embedding is deliberately outside the lock.
        vector = tuple(float(value) for value in embed())
        unit_vector = _unit_vector(vector)

        now = self._clock()
        with self._lock:
            self._purge_expired_locked(now)
            best_key: CacheKey | None = None
            best_similarity = -1.0
            for candidate_key, candidate in self._entries.items():
                if candidate.namespace != namespace:
                    continue
                similarity = _cosine(unit_vector, candidate.unit_vector)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_key = candidate_key

            if best_key is not None and best_similarity >= self._similarity_threshold:
                entry = self._entries[best_key]
                self._entries.move_to_end(best_key)
                updated = replace(
                    entry,
                    last_accessed=now,
                    hits=entry.hits + 1,
                )
                self._entries[best_key] = updated
                self._semantic_hits += 1
                return CacheHit(
                    hit=True,
                    payload=updated.payload,
                    query_vector=vector,
                    similarity=best_similarity,
                    kind="semantic",
                    age_seconds=max(0.0, now - updated.created_at),
                )

            self._misses += 1
            return CacheHit(hit=False, query_vector=vector)

    # ------------------------------------------------------------------
    # Writes and maintenance
    # ------------------------------------------------------------------
    def put(
        self,
        query: str,
        vector: Sequence[float],
        payload: Payload,
        *,
        namespace: str = "",
    ) -> bool:
        """Store ``payload`` for ``query``/``vector``; return whether it was stored."""
        if self._max_entries <= 0:
            return False

        now = self._clock()
        normalized = normalize_query(query)
        key: CacheKey = (namespace, normalized)
        vector_tuple = tuple(float(value) for value in vector)

        with self._lock:
            self._purge_expired_locked(now)
            # Replacing an existing key resets its TTL and refreshes its LRU
            # position, which is the desired behaviour for a changed payload.
            self._entries.pop(key, None)
            self._entries[key] = CacheEntry(
                query=query,
                normalized_query=normalized,
                vector=vector_tuple,
                unit_vector=_unit_vector(vector_tuple),
                payload=payload,
                namespace=namespace,
                created_at=now,
                last_accessed=now,
            )
            self._stores += 1

            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

        return True

    def clear(self, namespace: str | None = None) -> int:
        """Remove every entry, or every entry in ``namespace``; return the count."""
        with self._lock:
            if namespace is None:
                removed = len(self._entries)
                self._entries.clear()
                return removed

            keys = [
                key
                for key, entry in self._entries.items()
                if entry.namespace == namespace
            ]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def prune(self) -> int:
        """Drop expired entries and return how many were removed."""
        with self._lock:
            return self._purge_expired_locked(self._clock())

    def stats(self) -> CacheStats:
        """Return a consistent snapshot of the cache counters."""
        with self._lock:
            self._purge_expired_locked(self._clock())
            namespaces = {entry.namespace for entry in self._entries.values()}
            return CacheStats(
                entries=len(self._entries),
                namespaces=len(namespaces),
                capacity=self._max_entries,
                hits=self._exact_hits + self._semantic_hits,
                exact_hits=self._exact_hits,
                semantic_hits=self._semantic_hits,
                misses=self._misses,
                stores=self._stores,
                evictions=self._evictions,
                expirations=self._expirations,
                ttl_seconds=self._ttl_seconds,
                similarity_threshold=self._similarity_threshold,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _purge_expired_locked(self, now: float) -> int:
        """Remove expired entries. Caller must hold ``self._lock``."""
        if self._ttl_seconds <= 0:
            return 0

        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.created_at >= self._ttl_seconds
        ]
        for key in expired:
            del self._entries[key]
        self._expirations += len(expired)
        return len(expired)


__all__ = [
    "CacheEntry",
    "CacheHit",
    "CacheStats",
    "SemanticCache",
    "cache_namespace",
    "normalize_query",
]

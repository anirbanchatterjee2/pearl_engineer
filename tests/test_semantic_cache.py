"""Tests for the semantic cache."""

from __future__ import annotations

import pytest

from pearl_engineer.semantic_cache import (
    SemanticCache,
    cache_namespace,
    normalize_query,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_normalize_query_collapses_case_and_whitespace() -> None:
    assert normalize_query("  Kafka\tEvents \n") == "kafka events"


def test_exact_hit_does_not_call_embed() -> None:
    cache = SemanticCache()
    cache.put("kafka events", [1.0, 0.0], "payload")

    def embed() -> list[float]:
        raise AssertionError("embed must not be called for an exact hit")

    hit = cache.get("  KAFKA   events ", embed)

    assert hit.hit is True
    assert hit.kind == "exact"
    assert hit.payload == "payload"
    assert hit.similarity == 1.0
    assert hit.query_vector == (1.0, 0.0)


def test_semantic_hit_uses_cosine_similarity() -> None:
    cache = SemanticCache(similarity_threshold=0.9)
    cache.put("order events", [1.0, 0.0], "payload")

    hit = cache.get("order stream", lambda: [0.99, 0.01])

    assert hit.hit is True
    assert hit.kind == "semantic"
    assert hit.payload == "payload"
    assert hit.similarity == pytest.approx(0.99995, rel=1e-4)
    assert hit.query_vector == (0.99, 0.01)


def test_similarity_below_threshold_misses() -> None:
    cache = SemanticCache(similarity_threshold=0.9)
    cache.put("order events", [1.0, 0.0], "payload")

    hit = cache.get("billing invoices", lambda: [0.0, 1.0])

    assert hit.hit is False
    assert hit.payload is None
    assert hit.query_vector == (0.0, 1.0)


def test_entries_expire_after_ttl() -> None:
    clock = FakeClock()
    cache = SemanticCache(ttl_seconds=10.0, clock=clock)
    cache.put("kafka", [1.0, 0.0], "payload")

    clock.advance(9.0)
    assert cache.get("kafka", lambda: [1.0, 0.0]).hit is True

    clock.advance(2.0)
    hit = cache.get("kafka", lambda: [1.0, 0.0])

    assert hit.hit is False
    assert cache.stats().expirations == 1
    assert cache.stats().entries == 0


def test_zero_ttl_disables_expiry() -> None:
    clock = FakeClock()
    cache = SemanticCache(ttl_seconds=0.0, clock=clock)
    cache.put("kafka", [1.0, 0.0], "payload")

    clock.advance(10_000.0)
    assert cache.get("kafka", lambda: [1.0, 0.0]).hit is True
    assert cache.stats().expirations == 0


def test_lru_eviction_respects_capacity() -> None:
    cache = SemanticCache(max_entries=2, similarity_threshold=1.0)
    cache.put("a", [1.0, 0.0], "a")
    cache.put("b", [0.0, 1.0], "b")
    cache.get("a", lambda: [1.0, 0.0])  # refresh a
    cache.put("c", [1.0, 1.0], "c")

    assert cache.get("b", lambda: [0.0, 1.0]).hit is False
    assert cache.get("a", lambda: [1.0, 0.0]).hit is True
    assert cache.get("c", lambda: [1.0, 1.0]).hit is True
    assert cache.stats().evictions == 1


def test_namespaces_are_isolated() -> None:
    cache = SemanticCache(similarity_threshold=0.0)
    cache.put("kafka", [1.0, 0.0], "diagrams", namespace="diagrams|m|n=5")

    assert cache.get("kafka", lambda: [1.0, 0.0], namespace="other|m|n=5").hit is False
    assert (
        cache.get("kafka", lambda: [1.0, 0.0], namespace="diagrams|m|n=5").hit is True
    )


def test_clear_removes_one_namespace_or_everything() -> None:
    cache = SemanticCache(similarity_threshold=0.0)
    cache.put("a", [1.0, 0.0], "a", namespace="one")
    cache.put("b", [0.0, 1.0], "b", namespace="two")

    assert cache.clear("one") == 1
    assert len(cache) == 1
    assert cache.clear() == 1
    assert len(cache) == 0


def test_stats_track_hits_misses_and_stores() -> None:
    cache = SemanticCache(similarity_threshold=1.0)
    cache.put("a", [1.0, 0.0], "a")
    cache.get("a", lambda: [1.0, 0.0])
    cache.get("b", lambda: [0.0, 1.0])
    stats = cache.stats()

    assert stats.entries == 1
    assert stats.stores == 1
    assert stats.hits == 1
    assert stats.exact_hits == 1
    assert stats.misses == 1
    assert stats.hit_rate == 0.5
    assert stats.to_dict()["lookups"] == 2


def test_dimension_mismatch_never_matches() -> None:
    cache = SemanticCache(similarity_threshold=0.0)
    cache.put("a", [1.0, 0.0], "a")

    hit = cache.get("b", lambda: [1.0, 0.0, 0.0])

    assert hit.hit is False


def test_cache_namespace_includes_result_count() -> None:
    assert cache_namespace("diagrams", "model", 5) == "diagrams|model|n=5"
    assert cache_namespace("diagrams", "model", 0) == "diagrams|model|n=1"


def test_zero_capacity_does_not_store() -> None:
    cache = SemanticCache(max_entries=0)

    assert cache.put("a", [1.0, 0.0], "a") is False
    assert len(cache) == 0

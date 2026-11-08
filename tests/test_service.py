"""Tests for the shared retrieval service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pearl_engineer.semantic_cache import SemanticCache
from pearl_engineer.service import (
    CollectionUnavailableError,
    EmbeddingError,
    RetrievalError,
    SearchService,
)


class FakeCollection:
    def __init__(
        self,
        *,
        total: int = 1,
        query_result: dict[str, Any] | None = None,
        count_error: Exception | None = None,
        query_error: Exception | None = None,
    ) -> None:
        self.total = total
        self.count_error = count_error
        self.query_error = query_error
        self.query_result = query_result or {
            "documents": [["a diagram description"]],
            "metadatas": [[{"image_path": "C:/diagrams/a.png", "image_file": "a.png"}]],
            "distances": [[0.25]],
        }
        self.query_calls: list[dict[str, Any]] = []

    def count(self) -> int:
        if self.count_error is not None:
            raise self.count_error
        return self.total

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_calls.append(kwargs)
        if self.query_error is not None:
            raise self.query_error
        return self.query_result


class FakeEmbeddingClient:
    def __init__(
        self,
        vector: list[float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.vector = vector or [0.1, 0.2]
        self.error = error
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str]) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=self.vector)])


def make_service(
    collection: Any,
    embedding: Any | None = None,
    *,
    cache: SemanticCache | None = None,
) -> SearchService:
    return SearchService(
        collection=collection,
        embedding_client=embedding or FakeEmbeddingClient(),
        embedding_model="test-model",
        collection_name="diagrams",
        cache=cache,
    )


def test_search_returns_results_and_caches_them() -> None:
    collection = FakeCollection()
    embeddings = FakeEmbeddingClient()
    service = make_service(collection, embeddings, cache=SemanticCache())

    first = service.search("kafka", 5)
    second = service.search("  KAFKA ", 5)

    assert len(first.results) == 1
    assert first.results[0].image_file == "a.png"
    assert first.results[0].similarity == pytest.approx(0.75)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.cache_kind == "exact"
    assert embeddings.calls == 1
    assert len(collection.query_calls) == 1


def test_search_semantic_cache_hit() -> None:
    collection = FakeCollection()
    embeddings = FakeEmbeddingClient()
    service = make_service(collection, embeddings, cache=SemanticCache())

    service.search("order events", 5)
    outcome = service.search("order event stream", 5)

    assert outcome.cache_hit is True
    assert outcome.cache_kind == "semantic"
    assert outcome.cache_similarity == pytest.approx(1.0)
    assert embeddings.calls == 2
    assert len(collection.query_calls) == 1


def test_search_limits_results_to_collection_size() -> None:
    collection = FakeCollection(total=1)
    service = make_service(collection)

    service.search("kafka", 50)

    assert collection.query_calls[0]["n_results"] == 1


def test_search_without_collection_raises() -> None:
    service = make_service(None)

    with pytest.raises(CollectionUnavailableError):
        service.search("kafka")


def test_search_with_empty_collection_does_not_embed() -> None:
    collection = FakeCollection(total=0)
    embeddings = FakeEmbeddingClient()
    service = make_service(collection, embeddings, cache=SemanticCache())

    outcome = service.search("kafka")

    assert outcome.results == ()
    assert embeddings.calls == 0
    assert collection.query_calls == []


def test_embedding_failure_is_wrapped() -> None:
    service = make_service(
        FakeCollection(),
        FakeEmbeddingClient(error=RuntimeError("provider down")),
    )

    with pytest.raises(EmbeddingError, match="provider down"):
        service.search("kafka")


def test_query_failure_is_wrapped() -> None:
    service = make_service(FakeCollection(query_error=RuntimeError("index corrupt")))

    with pytest.raises(RetrievalError, match="index corrupt"):
        service.search("kafka")


def test_count_failure_is_wrapped() -> None:
    service = make_service(FakeCollection(count_error=RuntimeError("storage offline")))

    with pytest.raises(RetrievalError, match="storage offline"):
        service.search("kafka")


def test_cache_stats_and_clear_without_cache() -> None:
    service = make_service(FakeCollection(), cache=None)

    assert service.cache_stats() == {"enabled": False}
    assert service.clear_cache() == 0


def test_cache_stats_reports_hits() -> None:
    service = make_service(FakeCollection(), cache=SemanticCache())
    service.search("kafka")
    service.search("kafka")

    stats = service.cache_stats()

    assert stats["enabled"] is True
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1

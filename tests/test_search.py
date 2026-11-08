"""Tests for the FastAPI search layer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pearl_engineer.config import CacheSettings, SearchSettings
from pearl_engineer.search import create_app


class FakeCollection:
    def __init__(
        self, total: int = 1, query_result: dict[str, Any] | None = None
    ) -> None:
        self.total = total
        self.query_result = query_result or {
            "documents": [["a diagram description"]],
            "metadatas": [[{"image_path": "C:/diagrams/a.png", "image_file": "a.png"}]],
            "distances": [[0.25]],
        }
        self.last_query: dict[str, Any] | None = None

    def count(self) -> int:
        return self.total

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.last_query = kwargs
        return self.query_result


class FakeEmbeddingClient:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2]
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str]) -> object:
        self.calls += 1
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=self.vector)])


def make_client(
    collection: Any,
    *,
    embedding_client: FakeEmbeddingClient | None = None,
    cache: CacheSettings | None = None,
    admin_token: str = "",
) -> TestClient:
    settings = SearchSettings(
        persist_dir=Path("unused"),
        collection="diagrams",
        embedding_model="test-embedding-model",
        cache=cache or CacheSettings(),
        admin_token=admin_token,
    )
    app = create_app(
        settings,
        collection=collection,
        openai_client=embedding_client or FakeEmbeddingClient(),
    )
    return TestClient(app)


def test_search_returns_hits() -> None:
    collection = FakeCollection()

    with make_client(collection) as client:
        response = client.post("/search", json={"query": "kafka", "n_results": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "kafka"
    assert len(body["results"]) == 1
    assert body["results"][0]["description"] == "a diagram description"
    assert body["results"][0]["image_file"] == "a.png"
    assert body["results"][0]["similarity"] == pytest.approx(0.75)
    assert collection.last_query is not None
    assert collection.last_query["n_results"] == 1


def test_search_without_collection_returns_503() -> None:
    with make_client(None) as client:
        response = client.post("/search", json={"query": "kafka"})

    assert response.status_code == 503
    assert "not found" in response.json()["detail"]


def test_search_with_empty_collection_returns_no_results() -> None:
    with make_client(FakeCollection(total=0)) as client:
        response = client.post("/search", json={"query": "kafka"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_rejects_blank_query() -> None:
    with make_client(FakeCollection()) as client:
        response = client.post("/search", json={"query": "   "})

    assert response.status_code == 422


def test_health_reports_record_count() -> None:
    with make_client(FakeCollection(total=7)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["collection"] == "diagrams"
    assert body["records"] == 7
    assert body["cache"]["enabled"] is True
    assert body["cache"]["entries"] == 0


def test_search_exact_cache_hit_skips_embedding() -> None:
    embeddings = FakeEmbeddingClient()

    with make_client(FakeCollection(), embedding_client=embeddings) as client:
        first = client.post("/search", json={"query": "kafka"})
        second = client.post("/search", json={"query": "  KAFKA  "})

    assert first.status_code == 200
    assert first.headers["X-Cache"] == "MISS"
    assert second.status_code == 200
    assert second.headers["X-Cache"] == "HIT"
    assert second.headers["X-Cache-Kind"] == "exact"
    assert second.json()["cache"] == {
        "enabled": True,
        "hit": True,
        "kind": "exact",
        "similarity": 1.0,
        "age_seconds": second.json()["cache"]["age_seconds"],
    }
    assert embeddings.calls == 1


def test_search_semantic_cache_hit_for_equivalent_vector() -> None:
    embeddings = FakeEmbeddingClient()

    with make_client(FakeCollection(), embedding_client=embeddings) as client:
        client.post("/search", json={"query": "order events"})
        response = client.post("/search", json={"query": "order event stream"})

    assert response.status_code == 200
    assert response.headers["X-Cache"] == "HIT"
    assert response.headers["X-Cache-Kind"] == "semantic"
    assert response.json()["cache"]["kind"] == "semantic"
    assert response.json()["cache"]["similarity"] == pytest.approx(1.0)
    assert embeddings.calls == 2  # both queries must be embedded


def test_search_cache_can_be_disabled() -> None:
    embeddings = FakeEmbeddingClient()

    with make_client(
        FakeCollection(),
        embedding_client=embeddings,
        cache=CacheSettings(enabled=False),
    ) as client:
        client.post("/search", json={"query": "kafka"})
        response = client.post("/search", json={"query": "kafka"})

    assert response.headers["X-Cache"] == "MISS"
    assert response.json()["cache"]["enabled"] is False
    assert embeddings.calls == 2


def test_cache_stats_and_clear_endpoints() -> None:
    with make_client(FakeCollection()) as client:
        client.post("/search", json={"query": "kafka"})

        stats = client.get("/cache/stats")
        cleared = client.post("/cache/clear")
        after = client.get("/cache/stats")

    assert stats.status_code == 200
    assert stats.json()["enabled"] is True
    assert stats.json()["entries"] == 1
    assert stats.json()["misses"] == 1
    assert cleared.json() == {"cleared": 1, "namespace": None}
    assert after.json()["entries"] == 0


def test_cache_clear_requires_admin_token_when_configured() -> None:
    with make_client(FakeCollection(), admin_token="secret") as client:
        unauthorized = client.post("/cache/clear")
        authorized = client.post("/cache/clear", headers={"X-Admin-Token": "secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200

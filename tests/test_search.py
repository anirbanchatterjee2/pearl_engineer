"""Tests for the FastAPI search layer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pearl_engineer.config import SearchSettings
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
    def __init__(self) -> None:
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str]) -> object:
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
        )


def make_client(collection: Any) -> TestClient:
    settings = SearchSettings(
        persist_dir=Path("unused"),
        collection="diagrams",
        embedding_model="test-embedding-model",
    )
    app = create_app(
        settings,
        collection=collection,
        openai_client=FakeEmbeddingClient(),
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
    assert response.json() == {
        "status": "ok",
        "collection": "diagrams",
        "records": 7,
    }

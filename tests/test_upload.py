"""Tests for diagram upload, download and category filtering."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from pearl_engineer.config import CacheSettings, SearchSettings
from pearl_engineer.ingest import safe_filename, unique_path
from pearl_engineer.search import create_app


class FakeCollection:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.last_query: dict[str, Any] | None = None
        self.last_get: dict[str, Any] | None = None
        self.get_result: dict[str, Any] = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

    def count(self) -> int:
        # The canned query result represents one record, even before an upload.
        return max(1, len(self.upserts))

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.last_query = kwargs
        return {
            "documents": [["A gearbox assembly with a bearing tolerance callout."]],
            "metadatas": [
                [
                    {
                        "image_path": "C:/diagrams/gearbox.png",
                        "image_file": "gearbox.png",
                        "category": "mechanical",
                        "title": "Gearbox",
                    }
                ]
            ],
            "distances": [[0.2]],
        }

    def get(self, **kwargs: Any) -> dict[str, Any]:
        self.last_get = kwargs
        return self.get_result


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.last_prompt = ""
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._describe))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _describe(self, **kwargs: Any) -> object:
        self.last_prompt = kwargs["messages"][0]["content"][0]["text"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="A gearbox assembly with a bearing tolerance callout."
                    )
                )
            ]
        )

    def _embed(self, **kwargs: Any) -> object:
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
            usage=SimpleNamespace(total_tokens=4),
        )


def make_client(
    tmp_path: Path,
    *,
    collection: FakeCollection | None = None,
    max_upload_bytes: int = 1024 * 1024,
) -> tuple[TestClient, FakeCollection, FakeOpenAIClient]:
    collection = collection or FakeCollection()
    openai_client = FakeOpenAIClient()
    settings = SearchSettings(
        persist_dir=tmp_path / "chroma",
        collection="diagrams",
        embedding_model="test-embedding-model",
        cache=CacheSettings(enabled=False),
        admin_token="",
        diagrams_dir=tmp_path / "diagrams",
        descriptions_file=tmp_path / "descriptions.json",
        categories_file=tmp_path / "categories.json",
        vision_model="test-vision-model",
        max_upload_bytes=max_upload_bytes,
    )
    app = create_app(
        settings,
        collection=collection,
        openai_client=openai_client,
    )
    return TestClient(app), collection, openai_client


def upload(
    client: TestClient,
    *,
    filename: str = "gearbox.PNG",
    category: str = "mechanical",
    title: str | None = "Gearbox assembly",
) -> Any:
    data: dict[str, str] = {"category": category}
    if title is not None:
        data["title"] = title
    return client.post(
        "/diagrams",
        files={"file": (filename, b"\x89PNG\r\n\x1a\nfake", "image/png")},
        data=data,
    )


def test_upload_saves_describes_and_indexes(tmp_path: Path) -> None:
    client, collection, openai_client = make_client(tmp_path)

    with client:
        response = upload(client)

    assert response.status_code == 201
    body = response.json()
    diagram = body["diagram"]
    assert diagram["image_file"] == "gearbox.png"
    assert diagram["category"] == "mechanical"
    assert diagram["category_label"] == "Mechanical"
    assert diagram["title"] == "Gearbox assembly"
    assert diagram["download_url"] == "/diagrams/gearbox.png/download"

    saved = tmp_path / "diagrams" / "gearbox.png"
    assert saved.read_bytes() == b"\x89PNG\r\n\x1a\nfake"
    assert "mechanical engineering drawing" in openai_client.last_prompt

    assert len(collection.upserts) == 1
    metadata = collection.upserts[0]["metadatas"][0]
    assert metadata["category"] == "mechanical"
    assert metadata["title"] == "Gearbox assembly"
    assert collection.upserts[0]["ids"] == ["gearbox.png"]

    descriptions = json.loads((tmp_path / "descriptions.json").read_text("utf-8"))
    categories = json.loads((tmp_path / "categories.json").read_text("utf-8"))
    assert descriptions["gearbox.png"].startswith("A gearbox assembly")
    assert categories == {"gearbox.png": "mechanical"}


def test_upload_rejects_unknown_category(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    with client:
        response = upload(client, category="spaceship")

    assert response.status_code == 422
    assert "unknown category" in response.json()["detail"]


def test_upload_rejects_unsupported_file(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    with client:
        response = client.post(
            "/diagrams",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"category": "mechanical"},
        )

    assert response.status_code == 415


def test_upload_rejects_large_file(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, max_upload_bytes=4)

    with client:
        response = upload(client)

    assert response.status_code == 413


def test_upload_renames_duplicate_files(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    with client:
        first = upload(client)
        second = upload(client)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["diagram"]["image_file"] == "gearbox.png"
    assert second.json()["diagram"]["image_file"] == "gearbox-1.png"


def test_view_and_download_endpoints(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    with client:
        upload(client)
        view = client.get("/diagrams/gearbox.png")
        download = client.get("/diagrams/gearbox.png/download")

    assert view.status_code == 200
    assert view.content == b"\x89PNG\r\n\x1a\nfake"
    assert download.status_code == 200
    assert download.content == b"\x89PNG\r\n\x1a\nfake"
    assert "attachment" in download.headers["content-disposition"]


def test_missing_diagram_returns_404(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    with client:
        response = client.get("/diagrams/nope.png/download")

    assert response.status_code == 404


def test_search_can_filter_by_category(tmp_path: Path) -> None:
    client, collection, _ = make_client(tmp_path)

    with client:
        upload(client)
        response = client.post(
            "/search",
            json={"query": "gearbox", "category": "mechanical", "n_results": 5},
        )

    assert response.status_code == 200
    assert collection.last_query is not None
    assert collection.last_query["where"] == {"category": "mechanical"}
    hit = response.json()["results"][0]
    assert hit["category"] == "mechanical"
    assert hit["category_label"] == "Mechanical"
    assert hit["image_url"] == "/diagrams/gearbox.png"
    assert hit["download_url"] == "/diagrams/gearbox.png/download"


def test_search_rejects_unknown_category(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    with client:
        response = client.post(
            "/search", json={"query": "gearbox", "category": "spaceship"}
        )

    assert response.status_code == 422


def test_list_diagrams_filters_by_category(tmp_path: Path) -> None:
    collection = FakeCollection()
    collection.get_result = {
        "ids": ["gearbox.png"],
        "documents": ["A gearbox assembly."],
        "metadatas": [
            {
                "image_path": "C:/diagrams/gearbox.png",
                "image_file": "gearbox.png",
                "category": "mechanical",
                "title": "Gearbox",
                "indexed_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    client, _, _ = make_client(tmp_path, collection=collection)

    with client:
        response = client.get("/diagrams?category=mechanical&limit=10")

    assert response.status_code == 200
    assert collection.last_get is not None
    assert collection.last_get["where"] == {"category": "mechanical"}
    assert response.json()[0]["title"] == "Gearbox"


def test_categories_and_ui_are_served(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    with client:
        categories = client.get("/categories")
        page = client.get("/")

    assert categories.status_code == 200
    assert [item["id"] for item in categories.json()] == [
        "mechanical",
        "civil",
        "computer_flow",
    ]
    assert page.status_code == 200
    assert "Pearl Engineer" in page.text


def test_safe_filename_and_unique_path(tmp_path: Path) -> None:
    assert safe_filename("../../Gear Box!.PNG") == "Gear-Box.png"
    assert safe_filename("...") == "diagram"

    (tmp_path / "a.png").write_bytes(b"x")
    assert unique_path(tmp_path, "a.png").name == "a-1.png"

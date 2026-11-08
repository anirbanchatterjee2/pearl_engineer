"""Tests for the embedding/indexing stage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pearl_engineer.index import (
    DescriptionsFileError,
    chunked,
    get_collection,
    index_descriptions,
    load_descriptions,
    truncate_text,
)


class FakeEmbeddingClient:
    """Minimal stand-in for ``openai.OpenAI`` embeddings."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2]
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str]) -> object:
        self.calls += 1
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=self.vector)
                for index in range(len(input))
            ],
            usage=SimpleNamespace(total_tokens=len(input)),
        )


class FakeCollection:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def count(self) -> int:
        return sum(len(upsert["ids"]) for upsert in self.upserts)


class FakeChroma:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.deleted: list[str] = []

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def get_or_create_collection(self, **_kwargs: Any) -> FakeCollection:
        return self.collection


def test_load_descriptions_filters_invalid_entries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "descriptions.json"
    path.write_text(
        json.dumps({"a.png": "text", "b.png": 1, "c.png": "   "}),
        encoding="utf-8",
    )

    assert load_descriptions(path) == {"a.png": "text"}
    assert "not a string" in capsys.readouterr().err


def test_load_descriptions_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "descriptions.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DescriptionsFileError, match="not valid JSON"):
        load_descriptions(path)


def test_load_descriptions_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "descriptions.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(DescriptionsFileError, match="JSON object"):
        load_descriptions(path)


def test_truncate_text() -> None:
    assert truncate_text("abcdef", 10) == ("abcdef", False)
    assert truncate_text("abcdef", 3) == ("abc", True)
    assert truncate_text("abcdef", 0) == ("abcdef", False)


def test_chunked() -> None:
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_get_collection_falls_back_to_legacy_metadata() -> None:
    class LegacyChroma:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def get_or_create_collection(self, **kwargs: Any) -> str:
            self.calls.append(kwargs)
            if "configuration" in kwargs:
                raise TypeError("unsupported keyword")
            return "collection"

    client = LegacyChroma()

    assert get_collection(client, "diagrams") == "collection"
    assert client.calls[0]["configuration"] == {"hnsw": {"space": "cosine"}}
    assert client.calls[1]["metadata"] == {"hnsw:space": "cosine"}


def test_index_descriptions_upserts_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "descriptions.json"
    json_path.write_text(json.dumps({"a.png": "text"}), encoding="utf-8")
    collection = FakeCollection()
    chroma = FakeChroma(collection)
    monkeypatch.setattr(
        "pearl_engineer.index.chromadb.PersistentClient",
        lambda **_kwargs: chroma,
    )
    client = FakeEmbeddingClient()

    code = index_descriptions(
        json_path=json_path,
        diagrams_dir=tmp_path,
        persist_dir=tmp_path / "chroma_db",
        client=client,
    )

    assert code == 0
    assert client.calls == 1
    assert collection.upserts[0]["ids"] == ["a.png"]
    assert collection.upserts[0]["embeddings"] == [[0.1, 0.2]]
    assert collection.upserts[0]["documents"] == ["text"]
    metadata = collection.upserts[0]["metadatas"][0]
    assert metadata["image_file"] == "a.png"
    assert metadata["image_path"].endswith("a.png")
    assert metadata["image_exists"] is False


def test_index_descriptions_dry_run_does_not_need_a_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_path = tmp_path / "descriptions.json"
    json_path.write_text(json.dumps({"a.png": "text"}), encoding="utf-8")

    code = index_descriptions(
        json_path=json_path,
        diagrams_dir=tmp_path,
        persist_dir=tmp_path / "chroma_db",
        dry_run=True,
    )

    assert code == 0
    assert "Dry run: 1 record(s)" in capsys.readouterr().out

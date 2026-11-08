"""Tests for the image-description stage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pearl_engineer.describe import (
    describe_diagrams,
    encode_image,
    find_images,
    load_existing_descriptions,
    write_json_atomic,
)


class FakeVisionClient:
    """Minimal stand-in for ``openai.OpenAI`` that never touches the network."""

    def __init__(self, content: str = "a detailed description") -> None:
        self.content = content
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **_kwargs: object) -> object:
        self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.content))
            ]
        )


def test_find_images_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "nested").mkdir()

    assert [path.name for path in find_images(tmp_path)] == ["a.jpg", "b.PNG"]


def test_find_images_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        find_images(tmp_path / "missing")


def test_encode_image_returns_data_url(tmp_path: Path) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"abc")

    assert encode_image(image) == "data:image/png;base64,YWJj"


def test_encode_image_rejects_too_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"x")
    monkeypatch.setattr("pearl_engineer.describe.MAX_IMAGE_BYTES", 0)

    with pytest.raises(ValueError, match="above the 20 MB API limit"):
        encode_image(image)


def test_write_json_atomic_creates_parents(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "descriptions.json"

    write_json_atomic(output, {"a.png": "text"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"a.png": "text"}
    assert not output.with_name(output.name + ".tmp").exists()


def test_load_existing_descriptions_keeps_only_strings(tmp_path: Path) -> None:
    output = tmp_path / "descriptions.json"
    output.write_text(
        json.dumps({"a.png": "text", "b.png": 3}), encoding="utf-8"
    )

    assert load_existing_descriptions(output) == {"a.png": "text"}


def test_load_existing_descriptions_ignores_invalid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "descriptions.json"
    output.write_text("{not json", encoding="utf-8")

    assert load_existing_descriptions(output) == {}
    assert "starting fresh" in capsys.readouterr().err


def test_describe_diagrams_writes_every_result(tmp_path: Path) -> None:
    diagrams = tmp_path / "diagrams"
    diagrams.mkdir()
    (diagrams / "a.png").write_bytes(b"x")
    (diagrams / "b.jpg").write_bytes(b"x")
    output = tmp_path / "data" / "descriptions.json"
    client = FakeVisionClient()

    code = describe_diagrams(diagrams_dir=diagrams, output=output, client=client)

    assert code == 0
    assert client.calls == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "a.png": "a detailed description",
        "b.jpg": "a detailed description",
    }


def test_describe_diagrams_resume_skips_existing(tmp_path: Path) -> None:
    diagrams = tmp_path / "diagrams"
    diagrams.mkdir()
    (diagrams / "a.png").write_bytes(b"x")
    (diagrams / "b.png").write_bytes(b"x")
    output = tmp_path / "descriptions.json"
    output.write_text(json.dumps({"a.png": "old text"}), encoding="utf-8")
    client = FakeVisionClient(content="new text")

    code = describe_diagrams(
        diagrams_dir=diagrams, output=output, resume=True, client=client
    )

    assert code == 0
    assert client.calls == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "a.png": "old text",
        "b.png": "new text",
    }


def test_describe_diagrams_returns_error_for_empty_folder(tmp_path: Path) -> None:
    diagrams = tmp_path / "diagrams"
    diagrams.mkdir()

    code = describe_diagrams(diagrams_dir=diagrams, client=FakeVisionClient())

    assert code == 1

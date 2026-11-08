"""Ingest uploaded diagram images into the search index.

The web UI posts a file plus a category. This module validates the upload,
stores it in the diagrams folder, asks the vision model for a description,
embeds that description, and upserts the record into ChromaDB in one step. It
also keeps ``descriptions.json`` and ``categories.json`` in sync so the
offline ``pearl describe`` / ``pearl index`` workflow sees the same data.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .categories import load_category_map, normalize_category
from .describe import (
    IMAGE_SUFFIXES,
    MAX_IMAGE_BYTES,
    describe_image,
    load_existing_descriptions,
    prompt_for_category,
    write_json_atomic,
)
from .index import embed_batch

logger = logging.getLogger(__name__)

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]+")
_MULTIPLE_SPACES = re.compile(r"\s+")
_WRITE_LOCK = threading.Lock()


class IngestError(RuntimeError):
    """Base class for upload/ingestion failures."""


class UnsupportedFileError(IngestError):
    """The upload is not a supported image type."""


class UploadTooLargeError(IngestError):
    """The upload exceeds the configured size limit."""


class EmptyUploadError(IngestError):
    """The upload has no bytes."""


class IngestionFailedError(IngestError):
    """Description, embedding or indexing failed."""


@dataclass(frozen=True)
class IngestResult:
    """A diagram that was saved and indexed."""

    image_file: str
    image_path: str
    description: str
    category: str
    title: str | None
    indexed_at: str


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe version of ``filename``.

    Only the basename is kept, so ``../../secrets.png`` can never escape the
    diagrams folder. Characters outside a conservative allow-list are replaced
    with underscores.
    """
    base = Path(filename).name
    stem = Path(base).stem
    suffix = Path(base).suffix.lower()
    stem = _UNSAFE_FILENAME.sub("_", stem).strip(" ._-")
    stem = _MULTIPLE_SPACES.sub("-", stem)
    if not stem:
        stem = "diagram"
    return f"{stem}{suffix}"


def unique_path(directory: Path, filename: str) -> Path:
    """Return a path in ``directory`` that does not collide with an existing file."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def validate_upload(filename: str, data: bytes, max_bytes: int) -> str:
    """Validate an upload and return its safe filename."""
    safe = safe_filename(filename)
    if Path(safe).suffix not in IMAGE_SUFFIXES:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise UnsupportedFileError(
            f"unsupported file type {Path(safe).suffix or '(none)'}; "
            f"supported types: {supported}"
        )
    if not data:
        raise EmptyUploadError("the uploaded file is empty")
    if len(data) > max_bytes:
        raise UploadTooLargeError(
            f"file is {len(data) / 1_048_576:.1f} MB; the limit is "
            f"{max_bytes / 1_048_576:.1f} MB"
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise UploadTooLargeError(
            f"file is {len(data) / 1_048_576:.1f} MB; the vision API limit is "
            f"{MAX_IMAGE_BYTES / 1_048_576:.0f} MB"
        )
    return safe


class DiagramIngestor:
    """Save, describe and index one uploaded diagram."""

    def __init__(
        self,
        *,
        diagrams_dir: Path,
        descriptions_file: Path,
        categories_file: Path,
        collection: Any | None,
        openai_client: Any | None,
        embedding_model: str,
        vision_model: str,
        max_upload_bytes: int = MAX_IMAGE_BYTES,
    ) -> None:
        self.diagrams_dir = diagrams_dir
        self.descriptions_file = descriptions_file
        self.categories_file = categories_file
        self.collection = collection
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.vision_model = vision_model
        self.max_upload_bytes = max_upload_bytes

    def ingest(
        self,
        *,
        filename: str,
        data: bytes,
        category: str,
        title: str | None = None,
    ) -> IngestResult:
        """Validate, store, describe and index an uploaded diagram."""
        if self.collection is None:
            raise IngestionFailedError(
                "the diagram collection is not available; run the index command first"
            )
        if self.openai_client is None:
            raise IngestionFailedError("no vision or embedding client is configured")

        category_id = normalize_category(category)
        if category_id is None:
            raise IngestError(f"unknown diagram category: {category!r}")

        safe = validate_upload(filename, data, self.max_upload_bytes)
        self.diagrams_dir.mkdir(parents=True, exist_ok=True)
        target = unique_path(self.diagrams_dir, safe)
        target.write_bytes(data)

        clean_title = (title or "").strip() or None
        try:
            description = describe_image(
                self.openai_client,
                target,
                self.vision_model,
                prompt=prompt_for_category(category_id),
            )
            document = description
            body = f"{clean_title}\n\n{description}" if clean_title else description
            vectors, _ = embed_batch(self.openai_client, [body], self.embedding_model)
            indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            metadata: dict[str, Any] = {
                "image_path": str(target.resolve()),
                "image_file": target.name,
                "category": category_id,
                "title": clean_title or "",
                "image_exists": True,
                "description_chars": len(document),
                "embedding_model": self.embedding_model,
                "indexed_at": indexed_at,
                "source": "upload",
            }
            self.collection.upsert(
                ids=[target.name],
                embeddings=vectors,
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as exc:
            target.unlink(missing_ok=True)
            if isinstance(exc, IngestError):
                raise
            raise IngestionFailedError(
                f"could not describe or index {target.name}: {exc}"
            ) from exc

        self._record_sidecars(target.name, description, category_id)
        return IngestResult(
            image_file=target.name,
            image_path=str(target.resolve()),
            description=description,
            category=category_id,
            title=clean_title,
            indexed_at=indexed_at,
        )

    def _record_sidecars(
        self, image_file: str, description: str, category: str
    ) -> None:
        """Persist the description and category maps used by the CLI workflow."""
        with _WRITE_LOCK:
            try:
                descriptions = load_existing_descriptions(self.descriptions_file)
                descriptions[image_file] = description
                write_json_atomic(self.descriptions_file, descriptions)

                categories = load_category_map(self.categories_file)
                categories[image_file] = category
                write_json_atomic(self.categories_file, categories)
            except OSError as exc:  # indexing succeeded; sidecars are best effort
                logger.warning(
                    "indexed %s but could not update sidecar files: %s",
                    image_file,
                    exc,
                )


__all__ = [
    "DiagramIngestor",
    "EmptyUploadError",
    "IngestError",
    "IngestResult",
    "IngestionFailedError",
    "UnsupportedFileError",
    "UploadTooLargeError",
    "safe_filename",
    "unique_path",
    "validate_upload",
]

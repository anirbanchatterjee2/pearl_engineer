"""Embed diagram descriptions and index them in a local ChromaDB collection.

Reads the JSON produced by :mod:`pearl_engineer.describe` (a top-level object
mapping each image filename to its text description), embeds every description
with OpenAI's ``text-embedding-3-small`` model, and upserts the vectors into a
persistent ChromaDB collection. Every record's metadata carries the absolute
path of the source image, so a retrieval hit can always be traced back to a
file on disk.

IDs are the image filenames, so re-running is idempotent: changed descriptions
are overwritten rather than duplicated.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import chromadb
from openai import OpenAI, OpenAIError

from .config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DESCRIPTIONS_FILE,
    DEFAULT_DIAGRAMS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    require_api_key,
)

# text-embedding-3-small accepts at most 8191 tokens per input. Descriptions are
# capped at 4000 output tokens upstream, so this character budget (~4 chars/token)
# only bites on pathological input; we truncate rather than let the API reject it.
DEFAULT_MAX_CHARS = 30_000

# The embeddings endpoint accepts up to 2048 inputs per request; keep batches
# smaller so one bad description cannot fail a large chunk of the run.
DEFAULT_BATCH_SIZE = 64


class DescriptionsFileError(Exception):
    """Raised when the descriptions JSON cannot be read or is malformed."""


def load_descriptions(path: Path) -> dict[str, str]:
    """Load the filename -> description map, rejecting anything malformed."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise DescriptionsFileError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DescriptionsFileError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise DescriptionsFileError(
            f"{path} must contain a JSON object mapping filenames to descriptions"
        )

    descriptions: dict[str, str] = {}
    for name, text in payload.items():
        if not isinstance(text, str):
            print(
                f"warning: skipping {name}: description is not a string",
                file=sys.stderr,
            )
            continue
        if not text.strip():
            print(
                f"warning: skipping {name}: description is empty",
                file=sys.stderr,
            )
            continue
        descriptions[name] = text
    return descriptions


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Return ``(text, was_truncated)`` capped to the embedding input budget."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def embed_batch(
    client: Any, texts: list[str], model: str
) -> tuple[list[list[float]], int]:
    """Embed one batch of texts; return vectors in input order plus tokens used."""
    response = client.embeddings.create(model=model, input=texts)
    # The API documents input order, but sorting on index is cheap insurance.
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = [item.embedding for item in ordered]
    if len(vectors) != len(texts):
        raise ValueError(f"expected {len(texts)} embeddings, received {len(vectors)}")
    tokens = response.usage.total_tokens if response.usage else 0
    return vectors, tokens


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    """Yield ``items`` in consecutive chunks of at most ``size``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_collection(client: Any, name: str) -> Any:
    """Open (or create) the collection, preferring cosine distance.

    ChromaDB 1.x takes the HNSW settings via ``configuration``; 0.4/0.5 only
    understands the older ``metadata={"hnsw:space": ...}`` form.
    """
    try:
        return client.get_or_create_collection(
            name=name, configuration={"hnsw": {"space": "cosine"}}
        )
    except TypeError:
        return client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


def index_descriptions(
    *,
    json_path: Path = DEFAULT_DESCRIPTIONS_FILE,
    diagrams_dir: Path = DEFAULT_DIAGRAMS_DIR,
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_chars: int = DEFAULT_MAX_CHARS,
    reset: bool = False,
    dry_run: bool = False,
    client: Any | None = None,
) -> int:
    """Embed descriptions and upsert them into ChromaDB.

    Returns a process exit code: 0 on success, 1 when some records failed, and
    2 when the input or configuration is invalid.
    """
    descriptions = load_descriptions(json_path)
    if not descriptions:
        print(f"No descriptions to index in {json_path}", file=sys.stderr)
        return 1

    # Resolve image paths up front and warn about entries whose file is missing;
    # the path is still stored so the record stays traceable.
    image_paths: dict[str, Path] = {}
    missing: list[str] = []
    for name in descriptions:
        path = (diagrams_dir / name).resolve()
        image_paths[name] = path
        if not path.is_file():
            missing.append(name)

    if missing:
        print(
            f"warning: {len(missing)} image(s) not found under "
            f"{diagrams_dir.resolve()}; indexing them anyway:",
            file=sys.stderr,
        )
        for name in missing[:10]:
            print(f"  - {name}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)

    if dry_run:
        print(
            f"Dry run: {len(descriptions)} record(s) -> "
            f"collection '{collection_name}' in {persist_dir}"
        )
        for name, text in descriptions.items():
            body, truncated = truncate_text(text, max_chars)
            suffix = " [truncated]" if truncated else ""
            print(f"  {name}: {len(body):,} chars -> {image_paths[name]}{suffix}")
        return 0

    if client is None:
        require_api_key()
        client = OpenAI(max_retries=5, timeout=120.0)

    chroma = chromadb.PersistentClient(path=str(persist_dir))

    if reset:
        try:
            chroma.delete_collection(collection_name)
            print(f"Dropped existing collection '{collection_name}'")
        except Exception as exc:  # chroma raises version-specific not-found errors
            print(f"No existing collection '{collection_name}' to drop ({exc})")

    collection = get_collection(chroma, collection_name)

    names = list(descriptions)
    total = len(names)
    indexed = 0
    total_tokens = 0
    failures: list[tuple[str, str]] = []
    started = time.monotonic()

    for batch_number, batch_names in enumerate(
        chunked(names, max(1, batch_size)), start=1
    ):
        documents: list[str] = []
        bodies: list[str] = []
        metadatas: list[dict[str, Any]] = []
        batch_truncated: list[str] = []
        indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for name in batch_names:
            body, truncated = truncate_text(descriptions[name], max_chars)
            if truncated:
                batch_truncated.append(name)
            # Keep the full description as the stored document; only the vector is
            # built from the truncated body, since that is all the model accepts.
            documents.append(descriptions[name])
            bodies.append(body)
            metadatas.append(
                {
                    # The payload that makes a vector answerable: where the image lives.
                    "image_path": str(image_paths[name]),
                    "image_file": name,
                    "image_exists": image_paths[name].is_file(),
                    "description_chars": len(descriptions[name]),
                    "embedding_model": model,
                    "indexed_at": indexed_at,
                }
            )

        if batch_truncated:
            print(
                f"warning: truncated {len(batch_truncated)} description(s) to "
                f"{max_chars:,} chars: {', '.join(batch_truncated[:5])}"
                f"{' ...' if len(batch_truncated) > 5 else ''}",
                file=sys.stderr,
            )

        print(
            f"[batch {batch_number}] embedding {len(batch_names)} description(s)...",
            end="",
            flush=True,
        )
        t0 = time.monotonic()
        try:
            embeddings, tokens = embed_batch(client, bodies, model)
        except (OpenAIError, ValueError) as exc:
            print(f" failed ({exc})")
            failures.extend((name, str(exc)) for name in batch_names)
            continue

        total_tokens += tokens
        try:
            collection.upsert(
                ids=batch_names,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:  # surface storage errors in the summary, keep going
            print(f" failed to upsert ({exc})")
            failures.extend((name, f"upsert failed: {exc}") for name in batch_names)
            continue

        indexed += len(batch_names)
        print(f" upserted in {time.monotonic() - t0:.1f}s ({tokens:,} tokens)")

    elapsed = time.monotonic() - started
    print(
        f"\nIndexed {indexed}/{total} record(s) into '{collection_name}' "
        f"({persist_dir}) in {elapsed:.1f}s; {total_tokens:,} embedding tokens"
    )
    print(f"Collection now holds {collection.count():,} record(s)")

    if failures:
        print(f"{len(failures)} record(s) failed:", file=sys.stderr)
        for name, err in failures:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1

    return 0

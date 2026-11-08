"""FastAPI backend for semantic search over the indexed diagram collection."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import chromadb
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    SearchSettings,
    require_api_key,
)

DEFAULT_N_RESULTS = 5
MAX_N_RESULTS = 50
MAX_QUERY_CHARS = 10_000

# Sentinel so tests can inject ``collection=None`` to mean "collection missing"
# without it being confused with "not injected".
_UNSET = object()

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """Body of the ``POST /search`` request."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Natural-language description of the diagram you are looking for.",
        examples=["service that publishes order events to Kafka"],
    )
    n_results: int = Field(
        DEFAULT_N_RESULTS,
        ge=1,
        le=MAX_N_RESULTS,
        description="Maximum number of matches to return.",
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty or whitespace")
        return value


class SearchHit(BaseModel):
    """One matching diagram."""

    description: str = Field(..., description="Stored description of the matching diagram.")
    image_path: str = Field(
        ..., description="Absolute path to the image file, from metadata."
    )
    image_file: str | None = Field(None, description="Image filename, from metadata.")
    distance: float | None = Field(None, description="Cosine distance; lower is closer.")
    similarity: float | None = Field(
        None, description="1 - distance for cosine collections."
    )


class SearchResponse(BaseModel):
    """Response body of ``POST /search``."""

    query: str
    results: list[SearchHit]


def _open_collection(client: Any, name: str) -> Any | None:
    """Return the existing collection, or None if it has not been built yet."""
    try:
        return client.get_collection(name)
    except Exception as exc:  # chroma raises version-specific not-found errors
        logger.warning("could not open collection %r: %s", name, exc)
        return None


def _embed_query(client: Any, text: str, model: str) -> list[float]:
    """Embed a single query string with the configured embedding model."""
    response = client.embeddings.create(model=model, input=[text])
    if not response.data:
        raise ValueError("embedding API returned no vectors")
    return response.data[0].embedding


def create_app(
    settings: SearchSettings | None = None,
    *,
    collection: Any = _UNSET,
    openai_client: Any = _UNSET,
) -> FastAPI:
    """Build the FastAPI application.

    ``collection`` and ``openai_client`` are injection points for tests; when
    omitted, the lifespan opens ChromaDB and creates a real OpenAI client.
    """
    cfg = settings or SearchSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if collection is not _UNSET:
            app.state.collection = collection
            app.state.openai = None if openai_client is _UNSET else openai_client
            yield
            return

        require_api_key()
        app.state.chroma = chromadb.PersistentClient(path=str(cfg.persist_dir))
        app.state.collection = _open_collection(app.state.chroma, cfg.collection)
        if app.state.collection is None:
            logger.warning(
                "collection %r not found under %s; /search will return 503 until "
                "the index command has been run",
                cfg.collection,
                cfg.persist_dir.resolve(),
            )
        app.state.openai = (
            openai_client
            if openai_client is not _UNSET
            else OpenAI(max_retries=5, timeout=60.0)
        )
        yield

    app = FastAPI(
        title="Pearl Engineer Diagram Search",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Report whether the collection is available and how many records it holds."""
        collection_obj = app.state.collection
        records = collection_obj.count() if collection_obj is not None else 0
        return {
            "status": "ok" if collection_obj is not None else "degraded",
            "collection": cfg.collection,
            "records": records,
        }

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        """Embed the query and return the closest diagrams from ChromaDB."""
        collection_obj = app.state.collection
        if collection_obj is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Collection '{cfg.collection}' not found in "
                    f"{cfg.persist_dir.resolve()}; run the index command first."
                ),
            )

        total = collection_obj.count()
        if total == 0:
            return SearchResponse(query=request.query, results=[])

        try:
            vector = _embed_query(
                app.state.openai, request.query, cfg.embedding_model
            )
        except (OpenAIError, ValueError) as exc:
            raise HTTPException(
                status_code=502, detail=f"embedding request failed: {exc}"
            ) from exc

        try:
            raw = collection_obj.query(
                query_embeddings=[vector],
                n_results=min(request.n_results, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # surface storage errors as a 500, not a stack trace
            raise HTTPException(
                status_code=500, detail=f"similarity search failed: {exc}"
            ) from exc

        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            score = float(distance) if distance is not None else None
            image_file = metadata.get("image_file")
            hits.append(
                SearchHit(
                    description=document or "",
                    image_path=str(metadata.get("image_path", "")),
                    image_file=(
                        str(image_file) if image_file is not None else None
                    ),
                    distance=score,
                    similarity=(1.0 - score) if score is not None else None,
                )
            )

        return SearchResponse(query=request.query, results=hits)

    return app


# Uvicorn target: ``uvicorn pearl_engineer.search:app``.
app = create_app()

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_EMBEDDING_MODEL",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "app",
    "create_app",
]

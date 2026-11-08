"""FastAPI backend for semantic search over the indexed diagram collection."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .categories import (
    CATEGORIES,
    CATEGORY_IDS,
    category_label,
    normalize_category,
)
from .config import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    SearchSettings,
    require_api_key,
)
from .ingest import (
    DiagramIngestor,
    EmptyUploadError,
    IngestError,
    IngestionFailedError,
    IngestResult,
    UnsupportedFileError,
    UploadTooLargeError,
)
from .semantic_cache import SemanticCache
from .service import (
    CollectionUnavailableError,
    EmbeddingError,
    RetrievalError,
    SearchResult,
    SearchService,
)

DEFAULT_N_RESULTS = 5
MAX_N_RESULTS = 50
MAX_QUERY_CHARS = 10_000
MAX_LIST_RESULTS = 500
STATIC_DIR = Path(__file__).resolve().parent / "static"

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
    category: str | None = Field(
        None,
        description="Optional diagram kind: mechanical, civil or computer_flow.",
        examples=["computer_flow"],
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty or whitespace")
        return value

    @field_validator("category")
    @classmethod
    def _normalise_category(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = normalize_category(value)
        if normalized is None:
            raise ValueError(
                f"unknown category; choose one of: {', '.join(CATEGORY_IDS)}"
            )
        return normalized


class CacheInfo(BaseModel):
    """Cache diagnostics attached to every search response."""

    enabled: bool = Field(..., description="Whether semantic caching is enabled.")
    hit: bool = Field(..., description="Whether this response came from the cache.")
    kind: str | None = Field(
        None, description="'exact' or 'semantic' when the lookup hit."
    )
    similarity: float | None = Field(
        None, description="Query similarity for a semantic cache hit."
    )
    age_seconds: float | None = Field(
        None, description="Age of the cached entry in seconds."
    )


class SearchHit(BaseModel):
    """One matching diagram."""

    description: str = Field(
        ..., description="Stored description of the matching diagram."
    )
    image_path: str = Field(
        ..., description="Absolute path to the image file, from metadata."
    )
    image_file: str | None = Field(None, description="Image filename, from metadata.")
    distance: float | None = Field(
        None, description="Cosine distance; lower is closer."
    )
    similarity: float | None = Field(
        None, description="1 - distance for cosine collections."
    )
    category: str | None = Field(None, description="Diagram kind, from metadata.")
    category_label: str | None = Field(None, description="Human-readable category.")
    title: str | None = Field(None, description="Optional title supplied on upload.")
    image_url: str | None = Field(None, description="Inline preview URL.")
    download_url: str | None = Field(None, description="Attachment download URL.")


class SearchResponse(BaseModel):
    """Response body of ``POST /search``."""

    query: str
    results: list[SearchHit]
    cache: CacheInfo


class CategoryInfo(BaseModel):
    """One diagram category exposed to the UI."""

    id: str
    label: str
    description: str


class DiagramRecord(BaseModel):
    """A stored diagram returned by the upload and listing endpoints."""

    image_file: str
    image_path: str
    description: str = ""
    category: str = "uncategorized"
    category_label: str = "Uncategorized"
    title: str | None = None
    indexed_at: str | None = None
    image_url: str
    download_url: str


class UploadResponse(BaseModel):
    """Response body of ``POST /diagrams``."""

    message: str
    diagram: DiagramRecord


def _open_collection(client: Any, name: str) -> Any | None:
    """Return the existing collection, or None if it has not been built yet."""
    try:
        return client.get_collection(name)
    except Exception as exc:  # chroma raises version-specific not-found errors
        logger.warning("could not open collection %r: %s", name, exc)
        return None


def _image_url(filename: str | None) -> str | None:
    if not filename:
        return None
    return f"/diagrams/{quote(filename, safe='')}"


def _download_url(filename: str | None) -> str | None:
    if not filename:
        return None
    return f"/diagrams/{quote(filename, safe='')}/download"


def _to_hit(result: SearchResult) -> SearchHit:
    return SearchHit(
        description=result.description,
        image_path=result.image_path,
        image_file=result.image_file,
        distance=result.distance,
        similarity=result.similarity,
        category=result.category,
        category_label=category_label(result.category),
        title=result.title,
        image_url=_image_url(result.image_file),
        download_url=_download_url(result.image_file),
    )


def _to_record(
    *,
    image_file: str,
    image_path: str,
    description: str,
    category: str,
    title: str | None,
    indexed_at: str | None,
) -> DiagramRecord:
    return DiagramRecord(
        image_file=image_file,
        image_path=image_path,
        description=description,
        category=category,
        category_label=category_label(category),
        title=title,
        indexed_at=indexed_at,
        image_url=_image_url(image_file) or "",
        download_url=_download_url(image_file) or "",
    )


def _record_from_result(result: IngestResult) -> DiagramRecord:
    return _to_record(
        image_file=result.image_file,
        image_path=result.image_path,
        description=result.description,
        category=result.category,
        title=result.title,
        indexed_at=result.indexed_at,
    )


def _records_from_get(raw: dict[str, Any]) -> list[DiagramRecord]:
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    records: list[DiagramRecord] = []
    for image_id, document, metadata in zip(ids, documents, metadatas, strict=False):
        metadata = metadata or {}
        image_file = str(metadata.get("image_file") or image_id)
        title = metadata.get("title")
        records.append(
            _to_record(
                image_file=image_file,
                image_path=str(metadata.get("image_path", "")),
                description=document or "",
                category=str(metadata.get("category") or "uncategorized"),
                title=str(title) if title else None,
                indexed_at=metadata.get("indexed_at"),
            )
        )
    records.sort(
        key=lambda record: (record.indexed_at or "", record.image_file), reverse=True
    )
    return records


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
    cache_cfg = cfg.cache.normalised()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if collection is not _UNSET:
            collection_obj = collection
            app.state.chroma = None
            app.state.openai = None if openai_client is _UNSET else openai_client
        else:
            import chromadb

            require_api_key()
            app.state.chroma = chromadb.PersistentClient(path=str(cfg.persist_dir))
            collection_obj = _open_collection(app.state.chroma, cfg.collection)
            app.state.openai = (
                openai_client
                if openai_client is not _UNSET
                else OpenAI(max_retries=5, timeout=60.0)
            )
            if collection_obj is None:
                logger.warning(
                    "collection %r not found under %s; /search will return 503 "
                    "until the index command has been run",
                    cfg.collection,
                    cfg.persist_dir.resolve(),
                )

        cache = (
            SemanticCache(
                similarity_threshold=cache_cfg.similarity_threshold,
                ttl_seconds=cache_cfg.ttl_seconds,
                max_entries=cache_cfg.max_entries,
            )
            if cache_cfg.enabled
            else None
        )
        app.state.collection = collection_obj
        app.state.cache = cache
        app.state.service = SearchService(
            collection=collection_obj,
            embedding_client=app.state.openai,
            embedding_model=cfg.embedding_model,
            collection_name=cfg.collection,
            cache=cache,
        )
        app.state.ingestor = DiagramIngestor(
            diagrams_dir=cfg.diagrams_dir,
            descriptions_file=cfg.descriptions_file,
            categories_file=cfg.categories_file,
            collection=collection_obj,
            openai_client=app.state.openai,
            embedding_model=cfg.embedding_model,
            vision_model=cfg.vision_model,
            max_upload_bytes=cfg.max_upload_bytes,
        )
        yield

    app = FastAPI(
        title="Pearl Engineer Diagram Search",
        version=__version__,
        lifespan=lifespan,
    )

    def _require_admin(token: str | None) -> None:
        if cfg.admin_token and token != cfg.admin_token:
            raise HTTPException(
                status_code=401,
                detail="invalid or missing admin token",
            )

    def _ensure_collection() -> Any | None:
        """Open or create the collection on first upload."""
        current = app.state.collection
        if current is not None:
            return current
        if app.state.chroma is None:
            return None
        try:
            from .index import get_collection

            current = get_collection(app.state.chroma, cfg.collection)
        except Exception as exc:
            logger.warning("could not create collection %r: %s", cfg.collection, exc)
            return None
        app.state.collection = current
        app.state.service.collection = current
        app.state.ingestor.collection = current
        return current

    def _diagram_path(filename: str) -> Path | None:
        """Resolve a filename inside the diagrams folder, rejecting escapes."""
        base = cfg.diagrams_dir.resolve()
        candidate = (base / filename).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return None
        return candidate

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Report collection availability, record count and cache metrics."""
        service: SearchService = app.state.service
        try:
            records = service.record_count()
            status = "ok" if service.collection is not None else "degraded"
        except RetrievalError as exc:
            logger.warning("health check could not count records: %s", exc)
            records = 0
            status = "degraded"

        return {
            "status": status,
            "collection": cfg.collection,
            "records": records,
            "cache": service.cache_stats(),
        }

    @app.get("/categories", response_model=list[CategoryInfo])
    def categories() -> list[CategoryInfo]:
        """List the diagram kinds understood by the product."""
        return [
            CategoryInfo(
                id=category.id,
                label=category.label,
                description=category.description,
            )
            for category in CATEGORIES
        ]

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest, response: Response) -> SearchResponse:
        """Embed the query and return the closest diagrams from ChromaDB."""
        service: SearchService = app.state.service
        try:
            outcome = service.search(
                request.query,
                request.n_results,
                category=request.category,
            )
        except CollectionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(
                status_code=502, detail=f"embedding request failed: {exc}"
            ) from exc
        except RetrievalError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        response.headers["X-Cache"] = "HIT" if outcome.cache_hit else "MISS"
        if outcome.cache_kind is not None:
            response.headers["X-Cache-Kind"] = outcome.cache_kind
        if outcome.cache_similarity is not None:
            response.headers["X-Cache-Similarity"] = f"{outcome.cache_similarity:.4f}"
        if outcome.cache_age_seconds is not None:
            response.headers["X-Cache-Age"] = f"{outcome.cache_age_seconds:.1f}"

        return SearchResponse(
            query=request.query,
            results=[_to_hit(result) for result in outcome.results],
            cache=CacheInfo(
                enabled=cache_cfg.enabled,
                hit=outcome.cache_hit,
                kind=outcome.cache_kind,
                similarity=outcome.cache_similarity,
                age_seconds=outcome.cache_age_seconds,
            ),
        )

    @app.get("/cache/stats")
    def cache_stats() -> dict[str, Any]:
        """Expose cache counters for monitoring and dashboards."""
        service: SearchService = app.state.service
        return service.cache_stats()

    @app.post("/cache/clear")
    def clear_cache(
        namespace: str | None = None,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        """Clear the semantic cache, optionally only one namespace."""
        _require_admin(x_admin_token)
        service: SearchService = app.state.service
        removed = service.clear_cache(namespace)
        return {"cleared": removed, "namespace": namespace}

    @app.get("/diagrams", response_model=list[DiagramRecord])
    def list_diagrams(
        category: str | None = None,
        limit: int = 100,
    ) -> list[DiagramRecord]:
        """List indexed diagrams, optionally filtered by category."""
        normalized = None
        if category is not None and category.strip():
            normalized = normalize_category(category)
            if normalized is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown category; choose one of: {', '.join(CATEGORY_IDS)}",
                )

        collection_obj = app.state.collection
        if collection_obj is None:
            return []

        kwargs: dict[str, Any] = {
            "include": ["documents", "metadatas"],
            "limit": max(1, min(limit, MAX_LIST_RESULTS)),
        }
        if normalized is not None:
            kwargs["where"] = {"category": normalized}
        try:
            raw = collection_obj.get(**kwargs)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"could not list diagrams: {exc}"
            ) from exc
        return _records_from_get(raw)

    @app.post("/diagrams", response_model=UploadResponse, status_code=201)
    def upload_diagram(
        file: UploadFile = File(..., description="PNG or JPEG diagram image."),
        category: str = Form(..., description="mechanical, civil or computer_flow."),
        title: str | None = Form(None, description="Optional short title."),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> UploadResponse:
        """Upload, describe and index a diagram in one request."""
        _require_admin(x_admin_token)
        category_id = normalize_category(category)
        if category_id is None:
            raise HTTPException(
                status_code=422,
                detail=f"unknown category; choose one of: {', '.join(CATEGORY_IDS)}",
            )

        collection_obj = _ensure_collection()
        if collection_obj is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Collection '{cfg.collection}' is not available; "
                    "run the index command or check ChromaDB."
                ),
            )

        data = file.file.read(cfg.max_upload_bytes + 1)
        try:
            result = app.state.ingestor.ingest(
                filename=file.filename or "diagram",
                data=data,
                category=category_id,
                title=title,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UnsupportedFileError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except EmptyUploadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except IngestionFailedError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return UploadResponse(
            message=f"{result.image_file} uploaded and indexed",
            diagram=_record_from_result(result),
        )

    @app.get("/diagrams/{filename}/download")
    def download_diagram(filename: str) -> FileResponse:
        """Download one diagram as an attachment."""
        path = _diagram_path(filename)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="diagram not found")
        return FileResponse(path, filename=path.name)

    @app.get("/diagrams/{filename}")
    def view_diagram(filename: str) -> FileResponse:
        """Return a diagram image for inline preview."""
        path = _diagram_path(filename)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="diagram not found")
        return FileResponse(path)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the single-page web UI."""
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="web UI is not installed")
        return FileResponse(page)

    return app


# Uvicorn target: ``uvicorn pearl_engineer.search:app``.
app = create_app()

__all__ = [
    "CacheInfo",
    "CategoryInfo",
    "DEFAULT_COLLECTION",
    "DEFAULT_EMBEDDING_MODEL",
    "DiagramRecord",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "UploadResponse",
    "app",
    "create_app",
]

"""Retrieval service shared by the HTTP API, LangChain and LangGraph.

Keeping the retrieval workflow here means every surface (``POST /search``, a
LangChain retriever, a LangGraph node, the ``pearl ask`` command) gets identical
caching, error handling and result parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .semantic_cache import SemanticCache, cache_namespace


class SearchServiceError(RuntimeError):
    """Base class for retrieval failures."""


class CollectionUnavailableError(SearchServiceError):
    """The ChromaDB collection has not been built yet."""


class EmbeddingError(SearchServiceError):
    """The embedding provider rejected the query or returned no vector."""


class RetrievalError(SearchServiceError):
    """ChromaDB failed to answer a similarity query."""


@dataclass(frozen=True)
class SearchResult:
    """One matching diagram, independent of the HTTP representation."""

    description: str
    image_path: str
    image_file: str | None
    distance: float | None
    similarity: float | None
    # Full ChromaDB metadata, including fields added by other ingestion paths
    # such as ``category``, ``title`` and ``source``.
    metadata: dict[str, Any] = field(default_factory=dict)
    category: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class SearchOutcome:
    """A retrieval response plus cache diagnostics."""

    query: str
    results: tuple[SearchResult, ...]
    cache_hit: bool = False
    cache_kind: str | None = None
    cache_similarity: float | None = None
    cache_age_seconds: float | None = None


class SearchService:
    """Retrieve diagrams from a ChromaDB collection with semantic caching."""

    def __init__(
        self,
        *,
        collection: Any | None,
        embedding_client: Any | None,
        embedding_model: str,
        collection_name: str,
        cache: SemanticCache | None = None,
    ) -> None:
        self.collection = collection
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        n_results: int = 5,
        *,
        category: str | None = None,
    ) -> SearchOutcome:
        """Return the closest diagrams for ``query``.

        ``category`` optionally restricts the query to one diagram kind. Raises
        :class:`CollectionUnavailableError`, :class:`EmbeddingError` or
        :class:`RetrievalError` so each caller can map failures to its own
        transport (HTTP status, tool error, graph error).
        """
        if self.collection is None:
            raise CollectionUnavailableError(
                f"Collection '{self.collection_name}' not found or not built; "
                "run the index command first."
            )

        total = self.record_count()
        if total == 0:
            return SearchOutcome(query=query, results=())

        n_results = max(1, int(n_results))
        namespace = cache_namespace(
            self.collection_name, self.embedding_model, n_results
        )
        if category:
            namespace = f"{namespace}|category={category}"

        if self.cache is not None:
            cached = self.cache.get(
                query,
                lambda: self._embed(query),
                namespace=namespace,
            )
            if cached.hit:
                results = tuple(cached.payload or ())
                return SearchOutcome(
                    query=query,
                    results=results,
                    cache_hit=True,
                    cache_kind=cached.kind,
                    cache_similarity=cached.similarity,
                    cache_age_seconds=cached.age_seconds,
                )
            vector = cached.query_vector
        else:
            vector = self._embed(query)

        if vector is None:  # pragma: no cover - defensive; get() always sets it
            raise EmbeddingError("cache returned no query vector")

        raw = self._query(vector, min(n_results, total), category=category)
        results = tuple(self._parse(raw))

        if self.cache is not None:
            self.cache.put(query, vector, results, namespace=namespace)

        return SearchOutcome(query=query, results=results)

    def record_count(self) -> int:
        """Return the number of records in the collection (0 when unavailable)."""
        if self.collection is None:
            return 0
        try:
            return int(self.collection.count())
        except Exception as exc:  # chroma raises backend-specific errors
            raise RetrievalError(f"could not count collection records: {exc}") from exc

    def cache_stats(self) -> dict[str, Any]:
        """Return cache diagnostics for health checks and dashboards."""
        if self.cache is None:
            return {"enabled": False}
        return {"enabled": True, **self.cache.stats().to_dict()}

    def clear_cache(self, namespace: str | None = None) -> int:
        """Clear the semantic cache (all namespaces or one)."""
        if self.cache is None:
            return 0
        return self.cache.clear(namespace)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> list[float]:
        if self.embedding_client is None:
            raise EmbeddingError("no embedding client is configured")
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=[text],
            )
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc

        if not getattr(response, "data", None):
            raise EmbeddingError("embedding API returned no vectors")
        return list(response.data[0].embedding)

    def _query(
        self,
        vector: list[float],
        n_results: int,
        *,
        category: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if category:
            kwargs["where"] = {"category": category}
        try:
            return self.collection.query(**kwargs)
        except Exception as exc:
            raise RetrievalError(f"similarity search failed: {exc}") from exc

    @staticmethod
    def _parse(raw: dict[str, Any]) -> list[SearchResult]:
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for document, metadata, distance in zip(
            documents, metadatas, distances, strict=False
        ):
            metadata = dict(metadata or {})
            score = float(distance) if distance is not None else None
            image_file = metadata.get("image_file")
            title = metadata.get("title")
            category = metadata.get("category")
            results.append(
                SearchResult(
                    description=document or "",
                    image_path=str(metadata.get("image_path", "")),
                    image_file=str(image_file) if image_file is not None else None,
                    distance=score,
                    similarity=(1.0 - score) if score is not None else None,
                    metadata=metadata,
                    category=str(category) if category else None,
                    title=str(title) if title else None,
                )
            )
        return results


__all__ = [
    "CollectionUnavailableError",
    "EmbeddingError",
    "RetrievalError",
    "SearchOutcome",
    "SearchResult",
    "SearchService",
    "SearchServiceError",
]

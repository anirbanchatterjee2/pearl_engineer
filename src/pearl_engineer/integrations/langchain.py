"""LangChain adapters for Pearl Engineer.

Exposes diagram retrieval as both a LangChain ``BaseRetriever`` and a
``BaseTool`` so the same service can be used by chains, agents and RAG
pipelines. Install the optional dependency group first::

    pip install "pearl-engineer[langchain]"

Example::

    from pearl_engineer.integrations.langchain import diagram_retriever

    retriever = diagram_retriever(service, n_results=5)
    documents = retriever.invoke("which service publishes order events?")
"""

from __future__ import annotations

import asyncio
from typing import Any

try:  # pragma: no cover - exercised only when the optional extra is installed
    from langchain_core.callbacks.manager import (
        CallbackManagerForRetrieverRun,
        CallbackManagerForToolRun,
    )
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover - depends on optional install
    raise ImportError(
        "The LangChain integration requires langchain-core. "
        'Install it with: pip install "pearl-engineer[langchain]"'
    ) from exc

from pydantic import BaseModel, Field

from ..service import SearchOutcome, SearchResult, SearchService

DEFAULT_TOOL_DESCRIPTION = (
    "Search the indexed architecture diagrams. Use this whenever the user asks "
    "about system components, services, databases, queues, data flows, protocols, "
    "deployment topology or other details that are visible in an architecture "
    "diagram. Returns the matching diagram descriptions and image paths."
)


class DiagramSearchInput(BaseModel):
    """Arguments accepted by :class:`DiagramSearchTool`."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language description of the diagram or architecture detail.",
    )
    n_results: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum number of diagrams to return.",
    )


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def format_outcome(
    outcome: SearchOutcome,
    *,
    max_description_chars: int = 4000,
) -> str:
    """Render a retrieval outcome as compact text for an LLM or agent."""
    if not outcome.results:
        return f"No matching diagrams found for {outcome.query!r}."

    lines = [
        f"Found {len(outcome.results)} diagram(s) for {outcome.query!r}.",
        f"cache: {'hit' if outcome.cache_hit else 'miss'}"
        + (f" ({outcome.cache_kind})" if outcome.cache_kind else ""),
    ]
    for position, result in enumerate(outcome.results, start=1):
        label = result.image_file or result.image_path or f"result-{position}"
        similarity = (
            f"{result.similarity:.3f}" if result.similarity is not None else "n/a"
        )
        lines.append(f"\n[{position}] {label} (similarity {similarity})")
        if result.image_path:
            lines.append(f"path: {result.image_path}")
        lines.append(_clip(result.description, max_description_chars))
    return "\n".join(lines)


def to_document(result: SearchResult) -> Document:
    """Convert a :class:`SearchResult` into a LangChain ``Document``.

    All ChromaDB metadata is preserved, so consumers can filter or cite by
    ``category``, ``title`` or any field added by another ingestion path.
    """
    metadata: dict[str, Any] = dict(result.metadata or {})
    metadata["image_path"] = result.image_path
    if result.image_file is not None:
        metadata["image_file"] = result.image_file
    if result.category is not None:
        metadata["category"] = result.category
    if result.title is not None:
        metadata["title"] = result.title
    if result.distance is not None:
        metadata["distance"] = result.distance
    if result.similarity is not None:
        metadata["similarity"] = result.similarity
    return Document(page_content=result.description, metadata=metadata)


class DiagramSearchTool(BaseTool):
    """LangChain tool that searches indexed architecture diagrams."""

    name: str = "search_architecture_diagrams"
    description: str = DEFAULT_TOOL_DESCRIPTION
    args_schema: type[BaseModel] = DiagramSearchInput
    service: Any = Field(exclude=True)
    n_results: int = Field(default=5, ge=1, le=50)
    max_description_chars: int = Field(default=4000, ge=0)

    def _run(
        self,
        query: str,
        n_results: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        outcome = self.service.search(query, n_results or self.n_results)
        return format_outcome(
            outcome,
            max_description_chars=self.max_description_chars,
        )

    async def _arun(
        self,
        query: str,
        n_results: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        # Retrieval is synchronous (ChromaDB + OpenAI); keep the event loop free.
        return await asyncio.to_thread(
            self._run,
            query,
            n_results,
            run_manager,
        )


class DiagramRetriever(BaseRetriever):
    """LangChain retriever backed by :class:`SearchService`."""

    service: Any = Field(exclude=True)
    n_results: int = Field(default=5, ge=1, le=50)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        outcome = self.service.search(query, self.n_results)
        return [to_document(result) for result in outcome.results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return await asyncio.to_thread(
            self._get_relevant_documents,
            query,
            run_manager=run_manager,
        )


def diagram_search_tool(
    service: SearchService,
    *,
    n_results: int = 5,
    max_description_chars: int = 4000,
) -> DiagramSearchTool:
    """Build a LangChain tool for ``service``."""
    return DiagramSearchTool(
        service=service,
        n_results=n_results,
        max_description_chars=max_description_chars,
    )


def diagram_retriever(
    service: SearchService,
    *,
    n_results: int = 5,
) -> DiagramRetriever:
    """Build a LangChain retriever for ``service``."""
    return DiagramRetriever(service=service, n_results=n_results)


__all__ = [
    "DiagramRetriever",
    "DiagramSearchInput",
    "DiagramSearchTool",
    "diagram_retriever",
    "diagram_search_tool",
    "format_outcome",
    "to_document",
]

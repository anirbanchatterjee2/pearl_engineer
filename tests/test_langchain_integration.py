"""Tests for the optional LangChain adapters."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from pearl_engineer.integrations.langchain import (  # noqa: E402
    diagram_retriever,
    diagram_search_tool,
    format_outcome,
)
from pearl_engineer.service import SearchOutcome, SearchResult  # noqa: E402


class FakeService:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = (
            results
            if results is not None
            else [
                SearchResult(
                    description="Order service publishes to Kafka.",
                    image_path="C:/diagrams/orders.png",
                    image_file="orders.png",
                    distance=0.2,
                    similarity=0.8,
                )
            ]
        )
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, n_results: int = 5) -> SearchOutcome:
        self.calls.append((query, n_results))
        return SearchOutcome(query=query, results=tuple(self.results))


def test_retriever_returns_documents_with_metadata() -> None:
    service = FakeService()

    documents = diagram_retriever(service, n_results=3).invoke("kafka")

    assert len(documents) == 1
    assert documents[0].page_content == "Order service publishes to Kafka."
    assert documents[0].metadata["image_file"] == "orders.png"
    assert documents[0].metadata["similarity"] == pytest.approx(0.8)
    assert service.calls == [("kafka", 3)]


def test_retriever_supports_async_invocation() -> None:
    import asyncio

    service = FakeService()

    documents = asyncio.run(diagram_retriever(service).ainvoke("kafka"))

    assert len(documents) == 1


def test_tool_returns_human_readable_results() -> None:
    service = FakeService()

    text = diagram_search_tool(service).invoke({"query": "kafka", "n_results": 2})

    assert "orders.png" in text
    assert "Order service publishes to Kafka." in text
    assert "similarity 0.800" in text
    assert service.calls == [("kafka", 2)]


def test_tool_handles_empty_results() -> None:
    service = FakeService(results=[])

    text = diagram_search_tool(service).invoke({"query": "nothing"})

    assert "No matching diagrams" in text


def test_tool_supports_async_invocation() -> None:
    import asyncio

    service = FakeService()

    text = asyncio.run(diagram_search_tool(service).ainvoke({"query": "kafka"}))

    assert "orders.png" in text


def test_format_outcome_clips_long_descriptions() -> None:
    outcome = SearchOutcome(
        query="q",
        results=(
            SearchResult(
                description="x" * 100,
                image_path="p",
                image_file="a.png",
                distance=0.0,
                similarity=1.0,
            ),
        ),
    )

    text = format_outcome(outcome, max_description_chars=10)

    assert "x" * 10 not in text
    assert text.endswith("…")

"""Tests for the optional LangGraph workflow."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from pearl_engineer.integrations.langgraph import (  # noqa: E402
    ask,
    build_diagram_graph,
)
from pearl_engineer.service import SearchOutcome, SearchResult  # noqa: E402


def make_result(name: str = "orders.png") -> SearchResult:
    return SearchResult(
        description="Order service publishes to Kafka.",
        image_path=f"C:/diagrams/{name}",
        image_file=name,
        distance=0.2,
        similarity=0.8,
    )


class FakeService:
    def __init__(self, responses: list[list[SearchResult]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, n_results: int = 5) -> SearchOutcome:
        self.calls.append((query, n_results))
        results = self.responses.pop(0) if self.responses else []
        return SearchOutcome(query=query, results=tuple(results))


class FakeLLM:
    def __init__(self, content: str = "Grounded answer [orders.png]") -> None:
        self.content = content
        self.calls = 0
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        self.calls += 1
        self.messages = messages
        return SimpleNamespace(content=self.content)


def test_graph_returns_deterministic_answer_without_llm() -> None:
    service = FakeService([[make_result()]])
    graph = build_diagram_graph(service)

    result = graph.invoke(
        {
            "query": "how are orders published?",
            "original_query": "how are orders published?",
            "refinements": 0,
        }
    )

    assert "orders.png" in result["answer"]
    assert "Order service publishes to Kafka." in result["answer"]
    assert result["sources"][0]["image_file"] == "orders.png"
    assert service.calls == [("how are orders published?", 5)]


def test_graph_uses_llm_when_supplied() -> None:
    service = FakeService([[make_result()]])
    llm = FakeLLM()
    graph = build_diagram_graph(service, llm=llm)

    result = graph.invoke(
        {
            "query": "how are orders published?",
            "original_query": "how are orders published?",
            "refinements": 0,
        }
    )

    assert result["answer"] == "Grounded answer [orders.png]"
    assert llm.calls == 1
    assert len(llm.messages) == 2
    assert "Order service publishes to Kafka." in llm.messages[1].content


def test_graph_reports_when_nothing_is_found() -> None:
    service = FakeService([[]])
    graph = build_diagram_graph(service, max_refinements=0)

    result = graph.invoke(
        {
            "query": "unknown",
            "original_query": "unknown",
            "refinements": 0,
        }
    )

    assert "No matching diagrams" in result["answer"]
    assert service.calls == [("unknown", 5)]


def test_graph_refines_query_when_first_search_is_empty() -> None:
    service = FakeService([[], [make_result("billing.png")]])
    graph = build_diagram_graph(service, max_refinements=1)

    result = graph.invoke(
        {
            "query": "billing",
            "original_query": "billing",
            "refinements": 0,
        }
    )

    assert "billing.png" in result["answer"]
    assert len(service.calls) == 2
    assert "architecture components" in service.calls[1][0]


def test_ask_helper_returns_final_state() -> None:
    service = FakeService([[make_result()]])

    result = ask("how are orders published?", service=service)

    assert result["answer"]
    assert result["sources"]

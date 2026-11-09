"""LangGraph workflow for question answering over architecture diagrams.

The compiled graph has three nodes:

``retrieve`` -> semantic search through :class:`SearchService`
``refine``   -> deterministic query expansion when nothing was found
``generate`` -> grounded answer from an optional LangChain chat model

Without an LLM the graph still returns a deterministic digest of the retrieved
descriptions, which makes it useful for smoke tests and offline evaluation.
Install the optional dependencies first::

    pip install "pearl-engineer[langgraph]"

Example::

    from langchain_openai import ChatOpenAI
    from pearl_engineer.integrations.langgraph import build_diagram_graph

    graph = build_diagram_graph(service, llm=ChatOpenAI(model="gpt-4o-mini"))
    result = graph.invoke(
        {"query": "how does billing consume order events?",
         "original_query": "how does billing consume order events?",
         "refinements": 0}
    )
    print(result["answer"])
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypedDict

try:  # pragma: no cover - exercised only when the optional extra is installed
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - depends on optional install
    raise ImportError(
        "The LangGraph integration requires langgraph and langchain-core. "
        'Install it with: pip install "pearl-engineer[langgraph]"'
    ) from exc

from ..config import DEFAULT_CHAT_MODEL
from ..service import SearchService
from .langchain import diagram_retriever, diagram_search_tool

DEFAULT_SYSTEM_PROMPT = """You are a senior software architect answering questions \
about an organisation's architecture diagrams. Use only the retrieved diagram \
descriptions below. Cite diagrams by their filename in square brackets, for \
example [orders.png]. If the retrieved material does not contain the answer, say \
so plainly instead of guessing."""

DEFAULT_AGENT_PROMPT = """You are Pearl, an architecture research assistant. Use the \
search_architecture_diagrams tool to find relevant diagrams before answering. \
Always cite the diagram filenames you relied on."""

DEFAULT_REFINEMENT_SUFFIX = (
    "architecture components services interfaces data flows protocols deployment"
)


class DiagramGraphState(TypedDict, total=False):
    """State exchanged between LangGraph nodes."""

    query: str
    original_query: str
    documents: list[Document]
    answer: str
    sources: list[dict[str, Any]]
    refinements: int
    cache_hit: bool
    error: str | None


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _message_text(message: Any) -> str:
    """Extract text from a LangChain message, including content blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def _sources(documents: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for document in documents:
        metadata = document.metadata or {}
        source = {
            key: metadata[key]
            for key in (
                "image_file",
                "image_path",
                "category",
                "title",
                "distance",
                "similarity",
            )
            if metadata.get(key) is not None
        }
        sources.append(source)
    return sources


def _format_documents(
    documents: list[Document],
    *,
    max_description_chars: int = 4000,
) -> str:
    lines = [f"Retrieved {len(documents)} diagram(s):"]
    for position, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        label = metadata.get("image_file") or metadata.get("image_path") or "unknown"
        lines.append(f"\n[{position}] {label}")
        lines.append(_clip(document.page_content, max_description_chars))
    return "\n".join(lines)


def build_diagram_graph(
    service: SearchService,
    *,
    llm: Any | None = None,
    n_results: int = 5,
    max_refinements: int = 1,
    system_prompt: str | None = None,
    query_refiner: Callable[[str, int], str] | None = None,
    checkpointer: Any | None = None,
    max_description_chars: int = 4000,
) -> Any:
    """Build and compile the retrieval/generation graph.

    ``llm`` is any LangChain chat model with an ``invoke`` method. When it is
    ``None`` the graph returns a deterministic digest instead of calling a
    model, which keeps tests and offline evaluation free of API calls.
    """
    retriever = diagram_retriever(service, n_results=n_results)
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    max_refinements = max(0, int(max_refinements))

    def retrieve(state: DiagramGraphState) -> dict[str, Any]:
        query = state.get("query") or state.get("original_query", "")
        try:
            documents = retriever.invoke(query)
        except Exception as exc:  # surface retrieval errors in state, not a traceback
            return {"documents": [], "sources": [], "error": str(exc)}
        return {
            "documents": documents,
            "sources": _sources(documents),
            "error": None,
        }

    def refine(state: DiagramGraphState) -> dict[str, Any]:
        refinements = int(state.get("refinements", 0)) + 1
        original = state.get("original_query") or state.get("query", "")
        if query_refiner is not None:
            refined = query_refiner(original, refinements)
        else:
            refined = f"{original} {DEFAULT_REFINEMENT_SUFFIX}".strip()
        return {"query": refined, "refinements": refinements}

    def route_after_retrieve(state: DiagramGraphState) -> str:
        if state.get("documents"):
            return "generate"
        if int(state.get("refinements", 0)) < max_refinements:
            return "refine"
        return "generate"

    def generate(state: DiagramGraphState) -> dict[str, Any]:
        documents = state.get("documents") or []
        if not documents:
            error = state.get("error")
            answer = "No matching diagrams were found in the index."
            if error:
                answer = f"{answer} Retrieval error: {error}"
            return {"answer": answer}

        if llm is None:
            return {
                "answer": _format_documents(
                    documents,
                    max_description_chars=max_description_chars,
                )
            }

        context = _format_documents(
            documents,
            max_description_chars=max_description_chars,
        )
        question = state.get("original_query") or state.get("query", "")
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Retrieved architecture diagrams:\n{context}"
                )
            ),
        ]
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            return {"answer": f"Answer generation failed: {exc}"}
        return {"answer": _message_text(response)}

    builder = StateGraph(DiagramGraphState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("refine", refine)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "refine": "refine"},
    )
    builder.add_edge("refine", "retrieve")
    builder.add_edge("generate", END)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return builder.compile(**compile_kwargs)


def ask(
    question: str,
    *,
    service: SearchService,
    llm: Any | None = None,
    n_results: int = 5,
    max_refinements: int = 1,
    system_prompt: str | None = None,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run one question through the graph and return the final state."""
    graph = build_diagram_graph(
        service,
        llm=llm,
        n_results=n_results,
        max_refinements=max_refinements,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
    config = (
        {"configurable": {"thread_id": thread_id}} if thread_id is not None else None
    )
    return graph.invoke(
        {
            "query": question,
            "original_query": question,
            "refinements": 0,
        },
        config=config,
    )


def build_react_agent(
    *,
    service: SearchService,
    llm: Any,
    n_results: int = 5,
    system_prompt: str | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build a tool-calling ReAct agent when LangGraph's prebuilt agent exists.

    The agent is convenient for conversational use; the explicit graph from
    :func:`build_diagram_graph` is easier to inspect, test and extend.
    """
    try:  # pragma: no cover - depends on the installed LangGraph version
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This LangGraph version does not expose create_react_agent; "
            "use build_diagram_graph instead."
        ) from exc

    tool = diagram_search_tool(service, n_results=n_results)
    kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    parameters = inspect.signature(create_react_agent).parameters
    prompt = system_prompt or DEFAULT_AGENT_PROMPT
    if "prompt" in parameters:
        kwargs["prompt"] = prompt
    elif "state_modifier" in parameters:  # older LangGraph releases
        kwargs["state_modifier"] = prompt

    return create_react_agent(llm, [tool], **kwargs)


def default_chat_model(
    model: str | None = None,
    *,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Create a ``ChatOpenAI`` using ``PEARL_CHAT_MODEL`` by default."""
    try:  # pragma: no cover - depends on the optional install
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langchain-openai is required for default_chat_model. "
            'Install it with: pip install "pearl-engineer[langgraph]"'
        ) from exc
    return ChatOpenAI(
        model=model or DEFAULT_CHAT_MODEL,
        temperature=temperature,
        **kwargs,
    )


__all__ = [
    "DEFAULT_AGENT_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "DiagramGraphState",
    "ask",
    "build_diagram_graph",
    "build_react_agent",
    "default_chat_model",
]

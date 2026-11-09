"""Optional integrations for agent frameworks.

These modules import their framework lazily so the core package has no hard
dependency on LangChain or LangGraph. Install the matching extra first::

    pip install "pearl-engineer[langchain]"
    pip install "pearl-engineer[langgraph]"
"""

from __future__ import annotations

__all__: list[str] = []

"""Diagram categories shared by the CLI, the HTTP API and the web UI.

Categories are stored as a string in each ChromaDB record's metadata so search
can be filtered without a second database. The canonical ids are stable; the
labels and descriptions are presentation details used by ``GET /categories``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNCATEGORIZED = "uncategorized"

_SEPARATORS = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class DiagramCategory:
    """One kind of diagram the product understands."""

    id: str
    label: str
    description: str


CATEGORIES: tuple[DiagramCategory, ...] = (
    DiagramCategory(
        id="mechanical",
        label="Mechanical",
        description=(
            "Machine parts, assemblies, dimensions, tolerances, materials and "
            "mechanical schematics."
        ),
    ),
    DiagramCategory(
        id="civil",
        label="Civil Engineering",
        description=(
            "Structural, site, utility, elevation and infrastructure drawings."
        ),
    ),
    DiagramCategory(
        id="computer_flow",
        label="Computer Flow",
        description=(
            "Software architecture, data-flow, process and computer science "
            "flow diagrams."
        ),
    ),
)

CATEGORY_IDS: tuple[str, ...] = tuple(category.id for category in CATEGORIES)
CATEGORY_BY_ID: dict[str, DiagramCategory] = {
    category.id: category for category in CATEGORIES
}

# Friendly spellings the UI, API clients or a spreadsheet export might send.
_ALIASES: dict[str, str] = {
    "mechanical": "mechanical",
    "mechanical_diagram": "mechanical",
    "mechanical_engineering": "mechanical",
    "mech": "mechanical",
    "civil": "civil",
    "civil_diagram": "civil",
    "civil_engineering": "civil",
    "civil_engineering_diagram": "civil",
    "structural": "civil",
    "structure": "civil",
    "computer": "computer_flow",
    "computer_flow": "computer_flow",
    "computer_flow_diagram": "computer_flow",
    "computer_science": "computer_flow",
    "flow": "computer_flow",
    "flowchart": "computer_flow",
    "flow_chart": "computer_flow",
    "software": "computer_flow",
    "software_architecture": "computer_flow",
    "data_flow": "computer_flow",
    "dfd": "computer_flow",
}


def normalize_category(value: str | None) -> str | None:
    """Return a canonical category id, or ``None`` when ``value`` is not valid.

    ``None`` and blank strings mean "no filter" rather than an error, which
    keeps optional query parameters convenient.
    """
    if value is None:
        return None
    key = _SEPARATORS.sub("_", value.strip().lower()).strip("_")
    if not key:
        return None
    return _ALIASES.get(key)


def category_label(value: str | None) -> str:
    """Return the human-readable label for a category id."""
    if value is None or value == UNCATEGORIZED:
        return "Uncategorized"
    category = CATEGORY_BY_ID.get(value)
    return category.label if category is not None else value


def load_category_map(path: Path) -> dict[str, str]:
    """Load an optional ``filename -> category`` JSON map.

    Missing or malformed files are treated as an empty map: categories are
    optional metadata, and a broken sidecar should not block indexing.
    """
    if not path.is_file():
        return {}
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    categories: dict[str, str] = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        normalized = normalize_category(value)
        if normalized is not None:
            categories[name] = normalized
    return categories


__all__ = [
    "CATEGORIES",
    "CATEGORY_BY_ID",
    "CATEGORY_IDS",
    "UNCATEGORIZED",
    "DiagramCategory",
    "category_label",
    "load_category_map",
    "normalize_category",
]

"""Central configuration and defaults for Pearl Engineer.

Every path and model name can be overridden with an environment variable so the
same code works from a checkout, a container, or an installed console script.
A ``.env`` file in the working directory is loaded automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Prefer a .env in the working directory (or one of its parents) over a file
# that happens to sit next to the installed package.
load_dotenv(find_dotenv(usecwd=True))


class ConfigError(RuntimeError):
    """Raised when required configuration (usually the API key) is missing."""


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


DEFAULT_DIAGRAMS_DIR = _env_path("PEARL_DIAGRAMS_DIR", Path("diagrams"))
DEFAULT_DATA_DIR = _env_path("PEARL_DATA_DIR", Path("data"))
DEFAULT_DESCRIPTIONS_FILE = _env_path(
    "PEARL_DESCRIPTIONS_FILE", DEFAULT_DATA_DIR / "descriptions.json"
)
DEFAULT_CHROMA_DIR = _env_path("CHROMA_PERSIST_DIR", DEFAULT_DATA_DIR / "chroma_db")
DEFAULT_COLLECTION = _env_str("CHROMA_COLLECTION", "diagrams")
DEFAULT_EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_VISION_MODEL = _env_str("PEARL_VISION_MODEL", "gpt-4o-mini")


@dataclass(frozen=True)
class SearchSettings:
    """Runtime settings for the search API."""

    persist_dir: Path = DEFAULT_CHROMA_DIR
    collection: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @classmethod
    def from_env(cls) -> "SearchSettings":
        """Build settings from the current environment."""
        return cls(
            persist_dir=_env_path("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR),
            collection=_env_str("CHROMA_COLLECTION", DEFAULT_COLLECTION),
            embedding_model=_env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )


def require_api_key() -> str:
    """Return ``OPENAI_API_KEY`` or raise a user-facing configuration error."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Set it in your environment or .env file."
        )
    return key

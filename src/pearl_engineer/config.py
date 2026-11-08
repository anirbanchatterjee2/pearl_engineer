"""Central configuration and defaults for Pearl Engineer.

Every path and model name can be overridden with an environment variable so the
same code works from a checkout, a container, or an installed console script.
A ``.env`` file in the working directory is loaded automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


DEFAULT_DIAGRAMS_DIR = _env_path("PEARL_DIAGRAMS_DIR", Path("diagrams"))
DEFAULT_DATA_DIR = _env_path("PEARL_DATA_DIR", Path("data"))
DEFAULT_DESCRIPTIONS_FILE = _env_path(
    "PEARL_DESCRIPTIONS_FILE", DEFAULT_DATA_DIR / "descriptions.json"
)
DEFAULT_CATEGORIES_FILE = _env_path(
    "PEARL_CATEGORIES_FILE", DEFAULT_DATA_DIR / "categories.json"
)
DEFAULT_CHROMA_DIR = _env_path("CHROMA_PERSIST_DIR", DEFAULT_DATA_DIR / "chroma_db")
DEFAULT_COLLECTION = _env_str("CHROMA_COLLECTION", "diagrams")
DEFAULT_EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_VISION_MODEL = _env_str("PEARL_VISION_MODEL", "gpt-4o-mini")
DEFAULT_CHAT_MODEL = _env_str("PEARL_CHAT_MODEL", "gpt-4o-mini")

DEFAULT_CACHE_ENABLED = _env_bool("PEARL_CACHE_ENABLED", True)
DEFAULT_CACHE_SIMILARITY_THRESHOLD = _env_float(
    "PEARL_CACHE_SIMILARITY_THRESHOLD", 0.92
)
DEFAULT_CACHE_TTL_SECONDS = _env_float("PEARL_CACHE_TTL_SECONDS", 3600.0)
DEFAULT_CACHE_MAX_ENTRIES = _env_int("PEARL_CACHE_MAX_ENTRIES", 1024)
DEFAULT_ADMIN_TOKEN = _env_str("PEARL_ADMIN_TOKEN", "")
DEFAULT_MAX_UPLOAD_BYTES = _env_int("PEARL_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)


@dataclass(frozen=True)
class CacheSettings:
    """Runtime settings for the semantic cache.

    ``ttl_seconds <= 0`` disables expiry, ``max_entries <= 0`` disables the
    cache, and the similarity threshold is clamped to ``[0, 1]``.
    """

    enabled: bool = DEFAULT_CACHE_ENABLED
    similarity_threshold: float = DEFAULT_CACHE_SIMILARITY_THRESHOLD
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    max_entries: int = DEFAULT_CACHE_MAX_ENTRIES

    @classmethod
    def from_env(cls) -> CacheSettings:
        """Build settings from the current environment."""
        return cls(
            enabled=_env_bool("PEARL_CACHE_ENABLED", DEFAULT_CACHE_ENABLED),
            similarity_threshold=_env_float(
                "PEARL_CACHE_SIMILARITY_THRESHOLD",
                DEFAULT_CACHE_SIMILARITY_THRESHOLD,
            ),
            ttl_seconds=_env_float(
                "PEARL_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS
            ),
            max_entries=_env_int("PEARL_CACHE_MAX_ENTRIES", DEFAULT_CACHE_MAX_ENTRIES),
        )

    def normalised(self) -> CacheSettings:
        """Return a copy with values clamped to a safe operating range."""
        return CacheSettings(
            enabled=self.enabled and self.max_entries > 0,
            similarity_threshold=min(1.0, max(0.0, self.similarity_threshold)),
            ttl_seconds=max(0.0, self.ttl_seconds),
            max_entries=max(0, self.max_entries),
        )


@dataclass(frozen=True)
class SearchSettings:
    """Runtime settings for the search API."""

    persist_dir: Path = DEFAULT_CHROMA_DIR
    collection: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    cache: CacheSettings = field(default_factory=CacheSettings)
    admin_token: str = DEFAULT_ADMIN_TOKEN
    diagrams_dir: Path = DEFAULT_DIAGRAMS_DIR
    descriptions_file: Path = DEFAULT_DESCRIPTIONS_FILE
    categories_file: Path = DEFAULT_CATEGORIES_FILE
    vision_model: str = DEFAULT_VISION_MODEL
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

    @classmethod
    def from_env(cls) -> SearchSettings:
        """Build settings from the current environment."""
        return cls(
            persist_dir=_env_path("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR),
            collection=_env_str("CHROMA_COLLECTION", DEFAULT_COLLECTION),
            embedding_model=_env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            cache=CacheSettings.from_env(),
            admin_token=_env_str("PEARL_ADMIN_TOKEN", DEFAULT_ADMIN_TOKEN),
            diagrams_dir=_env_path("PEARL_DIAGRAMS_DIR", DEFAULT_DIAGRAMS_DIR),
            descriptions_file=_env_path(
                "PEARL_DESCRIPTIONS_FILE", DEFAULT_DESCRIPTIONS_FILE
            ),
            categories_file=_env_path("PEARL_CATEGORIES_FILE", DEFAULT_CATEGORIES_FILE),
            vision_model=_env_str("PEARL_VISION_MODEL", DEFAULT_VISION_MODEL),
            max_upload_bytes=_env_int(
                "PEARL_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES
            ),
        )


def require_api_key() -> str:
    """Return ``OPENAI_API_KEY`` or raise a user-facing configuration error."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Set it in your environment or .env file."
        )
    return key

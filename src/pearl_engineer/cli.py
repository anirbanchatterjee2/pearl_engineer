"""Command-line entry points for Pearl Engineer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from . import __version__
from .config import (
    ConfigError,
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DESCRIPTIONS_FILE,
    DEFAULT_DIAGRAMS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VISION_MODEL,
    require_api_key,
)
from .describe import DEFAULT_MAX_TOKENS
from .index import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_CHARS,
    DescriptionsFileError,
    index_descriptions,
)

Command = Callable[[argparse.Namespace], int]


def _add_describe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--diagrams-dir",
        type=Path,
        default=DEFAULT_DIAGRAMS_DIR,
        help=f"folder containing the images (default: {DEFAULT_DIAGRAMS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DESCRIPTIONS_FILE,
        help=f"JSON file to write (default: {DEFAULT_DESCRIPTIONS_FILE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_VISION_MODEL,
        help=f"vision model (default: {DEFAULT_VISION_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"cap on the length of each description (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep descriptions already present in the output file and skip them",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first error instead of continuing with the remaining images",
    )


def _add_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_DESCRIPTIONS_FILE,
        help=f"descriptions produced by 'describe' (default: {DEFAULT_DESCRIPTIONS_FILE})",
    )
    parser.add_argument(
        "--diagrams-dir",
        type=Path,
        default=DEFAULT_DIAGRAMS_DIR,
        help=f"folder containing the source images (default: {DEFAULT_DIAGRAMS_DIR})",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help=f"ChromaDB storage folder (default: {DEFAULT_CHROMA_DIR})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"descriptions per embedding request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=(
            "truncate descriptions longer than this before embedding "
            f"(default: {DEFAULT_MAX_CHARS}; 0 disables)"
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the collection before indexing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the records that would be indexed without calling any API",
    )


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="restart the server when source files change",
    )


def _cmd_describe(args: argparse.Namespace) -> int:
    from .describe import describe_diagrams

    return describe_diagrams(
        diagrams_dir=args.diagrams_dir,
        output=args.output,
        model=args.model,
        max_tokens=args.max_tokens,
        resume=args.resume,
        fail_fast=args.fail_fast,
    )


def _cmd_index(args: argparse.Namespace) -> int:
    try:
        return index_descriptions(
            json_path=args.json,
            diagrams_dir=args.diagrams_dir,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            model=args.model,
            batch_size=args.batch_size,
            max_chars=args.max_chars,
            reset=args.reset,
            dry_run=args.dry_run,
        )
    except DescriptionsFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    # Fail with a clear message before handing control to uvicorn.
    require_api_key()

    if args.reload:
        uvicorn.run(
            "pearl_engineer.search:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        from .search import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``pearl`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="pearl",
        description="Describe, index, and search architecture diagrams.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser(
        "describe",
        help="describe diagram images with a vision model",
        description="Extract architecture descriptions from diagram images.",
    )
    _add_describe_args(describe)
    describe.set_defaults(func=_cmd_describe)

    index = subparsers.add_parser(
        "index",
        help="embed descriptions and store them in ChromaDB",
        description="Embed diagram descriptions and index them in ChromaDB.",
    )
    _add_index_args(index)
    index.set_defaults(func=_cmd_index)

    serve = subparsers.add_parser(
        "serve",
        help="run the FastAPI search API",
        description="Serve semantic diagram search over HTTP.",
    )
    _add_serve_args(serve)
    serve.set_defaults(func=_cmd_serve)

    return parser


def _run(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> int:
    args = parser.parse_args(argv)
    func: Command = args.func
    try:
        return int(func(args))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``pearl`` console script."""
    return _run(build_parser(), argv)


def describe_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the standalone ``pearl-describe`` script."""
    parser = argparse.ArgumentParser(
        prog="pearl-describe",
        description="Extract architecture descriptions from diagram images.",
    )
    _add_describe_args(parser)
    parser.set_defaults(func=_cmd_describe)
    return _run(parser, argv)


def index_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the standalone ``pearl-index`` script."""
    parser = argparse.ArgumentParser(
        prog="pearl-index",
        description="Embed diagram descriptions and index them in ChromaDB.",
    )
    _add_index_args(parser)
    parser.set_defaults(func=_cmd_index)
    return _run(parser, argv)


def serve_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the standalone ``pearl-serve`` script."""
    parser = argparse.ArgumentParser(
        prog="pearl-serve",
        description="Serve semantic diagram search over HTTP.",
    )
    _add_serve_args(parser)
    parser.set_defaults(func=_cmd_serve)
    return _run(parser, argv)

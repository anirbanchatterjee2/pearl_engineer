"""Command-line entry points for Pearl Engineer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .categories import load_category_map
from .config import (
    DEFAULT_CATEGORIES_FILE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DESCRIPTIONS_FILE,
    DEFAULT_DIAGRAMS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VISION_MODEL,
    CacheSettings,
    ConfigError,
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
        "--categories",
        type=Path,
        default=DEFAULT_CATEGORIES_FILE,
        help=(
            "optional JSON map of image filename to category, used to pick "
            f"the right vision prompt (default: {DEFAULT_CATEGORIES_FILE})"
        ),
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
        "--categories",
        type=Path,
        default=DEFAULT_CATEGORIES_FILE,
        help=(
            "optional JSON map of image filename to category "
            f"(default: {DEFAULT_CATEGORIES_FILE})"
        ),
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
    parser.add_argument(
        "--cache",
        dest="cache_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable the semantic cache (default: PEARL_CACHE_ENABLED)",
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=None,
        help="cosine similarity needed for a semantic cache hit (default: 0.92)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=None,
        help="cache entry lifetime in seconds; 0 disables expiry (default: 3600)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=None,
        help="maximum cached queries; LRU eviction beyond it (default: 1024)",
    )


def _add_ask_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "question",
        help="question to answer from the indexed diagrams",
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
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help=f"LangGraph answer model (default: {DEFAULT_CHAT_MODEL})",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="diagrams to retrieve (default: 5)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="return the retrieved descriptions without calling a chat model",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the final graph state as JSON",
    )


def _add_crawl_tenders_args(parser: argparse.ArgumentParser) -> None:
    from .config import DEFAULT_TENDERS_FILE

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TENDERS_FILE,
        help=f"tender store JSON file (default: {DEFAULT_TENDERS_FILE})",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="search queries (default: gravity-filter set)",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="portal ids to crawl (default: all — see tenders sources)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="seed realistic sample tenders without network access",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="only list stored tenders and print the gist summary",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of the human gist",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="per-request timeout in seconds (default: 12)",
    )


def _cmd_crawl_tenders(args: argparse.Namespace) -> int:
    """Crawl defence portals for gravity-filter tenders and print the gist."""
    from .tenders import TenderCrawler, TenderStore, summarize_tenders

    store = TenderStore(Path(args.output))
    if not args.list:
        crawler = TenderCrawler(store, timeout=args.timeout)
        report = crawler.crawl(
            queries=args.queries, source_ids=args.sources, demo=args.demo
        )
    else:
        from .tenders import CrawlReport

        report = CrawlReport()

    records = store.list(limit=5000)
    gist = summarize_tenders(records)

    if args.json:
        payload: dict[str, Any] = {
            "report": report.to_dict(),
            "summary": gist,
            "tenders": [record.to_dict() for record in records[:100]],
        }
        print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
        return 0

    print(gist["headline"])
    print(f"\n{gist['gist']}\n")
    if not args.list:
        print(
            f"Crawled {report.fetched} page(s) across "
            f"{len(report.sources_tried)} portal(s): "
            f"{report.found} found, {report.new} new, {report.updated} updated."
        )
        if report.errors:
            print("\nPortal errors (best-effort, run continues):")
            for error in report.errors[:12]:
                print(f"  - {error}")
            if len(report.errors) > 12:
                print(f"  ... and {len(report.errors) - 12} more")
    if records:
        print("\nAdvertisements for gravity filters (soonest closing first):")
        for record in records[:25]:
            due = f"closes {record.closing_date}" if record.closing_date else "date TBA"
            print(
                f"  - {record.title[:90]} | {record.buyer or 'buyer n/a'} "
                f"| {due} | {record.source_name}"
            )
        if len(records) > 25:
            print(f"  ... and {len(records) - 25} more in {args.output}")
        print(
            f"\nFull list stored in {args.output} — also served in the UI "
            "under the 'Tender Ads' tab."
        )
    else:
        print(
            "No tenders stored yet. Try --demo for offline sample data, or "
            "crawl without --list to search the portals."
        )
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    from .describe import describe_diagrams

    return describe_diagrams(
        diagrams_dir=args.diagrams_dir,
        output=args.output,
        model=args.model,
        max_tokens=args.max_tokens,
        resume=args.resume,
        fail_fast=args.fail_fast,
        categories=load_category_map(args.categories),
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
            categories=load_category_map(args.categories),
        )
    except DescriptionsFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _apply_cache_overrides(args: argparse.Namespace) -> None:
    """Map ``pearl serve`` cache flags onto the environment read by settings."""
    if args.cache_enabled is not None:
        os.environ["PEARL_CACHE_ENABLED"] = "true" if args.cache_enabled else "false"
    if args.cache_threshold is not None:
        os.environ["PEARL_CACHE_SIMILARITY_THRESHOLD"] = str(args.cache_threshold)
    if args.cache_ttl is not None:
        os.environ["PEARL_CACHE_TTL_SECONDS"] = str(args.cache_ttl)
    if args.cache_size is not None:
        os.environ["PEARL_CACHE_MAX_ENTRIES"] = str(args.cache_size)


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    # Fail with a clear message before handing control to uvicorn.
    require_api_key()
    _apply_cache_overrides(args)

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


def _cmd_ask(args: argparse.Namespace) -> int:
    try:
        from .integrations.langgraph import ask as run_graph
        from .integrations.langgraph import default_chat_model
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import chromadb
    from openai import OpenAI

    from .semantic_cache import SemanticCache
    from .service import SearchService

    require_api_key()

    chroma = chromadb.PersistentClient(path=str(args.persist_dir))
    try:
        collection = chroma.get_collection(args.collection)
    except Exception as exc:
        print(
            f"error: collection '{args.collection}' not found in "
            f"{args.persist_dir.resolve()}; run 'pearl index' first ({exc})",
            file=sys.stderr,
        )
        return 2

    cache_cfg = CacheSettings.from_env().normalised()
    cache = (
        SemanticCache(
            similarity_threshold=cache_cfg.similarity_threshold,
            ttl_seconds=cache_cfg.ttl_seconds,
            max_entries=cache_cfg.max_entries,
        )
        if cache_cfg.enabled
        else None
    )
    service = SearchService(
        collection=collection,
        embedding_client=OpenAI(max_retries=5, timeout=60.0),
        embedding_model=args.model,
        collection_name=args.collection,
        cache=cache,
    )

    llm = None
    if not args.no_llm:
        try:
            llm = default_chat_model(args.chat_model)
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    result = run_graph(
        args.question,
        service=service,
        llm=llm,
        n_results=args.n_results,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        return 0

    print(result.get("answer", ""))
    sources = result.get("sources") or []
    if sources:
        print("\nSources:")
        for source in sources:
            label = source.get("image_file") or source.get("image_path")
            print(f"  - {label}")
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

    ask = subparsers.add_parser(
        "ask",
        help="answer a question with the LangGraph workflow",
        description="Retrieve diagrams and generate a grounded answer.",
    )
    _add_ask_args(ask)
    ask.set_defaults(func=_cmd_ask)

    crawl = subparsers.add_parser(
        "crawl-tenders",
        help="crawl defence portals for gravity-filter tenders",
        description=(
            "Crawl GeM, CPPP/eProcure, DRDO, MES, MoD and aggregators for "
            "gravity / water filter tenders, then print the gist summary."
        ),
    )
    _add_crawl_tenders_args(crawl)
    crawl.set_defaults(func=_cmd_crawl_tenders)

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


def ask_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the standalone ``pearl-ask`` script."""
    parser = argparse.ArgumentParser(
        prog="pearl-ask",
        description="Retrieve diagrams and generate a grounded answer.",
    )
    _add_ask_args(parser)
    parser.set_defaults(func=_cmd_ask)
    return _run(parser, argv)


def crawl_tenders_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the standalone ``pearl-crawl-tenders`` script."""
    parser = argparse.ArgumentParser(
        prog="pearl-crawl-tenders",
        description="Crawl defence portals for gravity-filter tenders.",
    )
    _add_crawl_tenders_args(parser)
    parser.set_defaults(func=_cmd_crawl_tenders)
    return _run(parser, argv)

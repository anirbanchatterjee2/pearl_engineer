"""Turn diagram images into exhaustive text descriptions with a vision model."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from .config import (
    DEFAULT_DESCRIPTIONS_FILE,
    DEFAULT_DIAGRAMS_DIR,
    DEFAULT_VISION_MODEL,
    require_api_key,
)

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # OpenAI's per-image limit
DEFAULT_MAX_TOKENS = 4000

PROMPT = """You are analysing a technical architecture diagram. Describe it exhaustively \
as a single detailed text report so that it can be understood without seeing the image.

Cover, in this order:
1. Components: every box, node, service, database, queue, gateway, client, actor, \
external system or group/zone shown. Give its label, its apparent type/role, and any \
technology, version, port, instance count or configuration detail written on or next to it.
2. Text: transcribe every visible label, title, legend entry, annotation, note, number \
and identifier verbatim. Do not summarise or paraphrase text that is written on the diagram.
3. Data flows: every arrow/line/connector, with source component, target component, \
direction, the label on the connector (protocol, payload, operation, e.g. "HTTPS / REST", \
"Kafka topic: orders"), and any stated number, rate or ordering.
4. Structure: titles, grouping boundaries (VPCs, subnets, trust boundaries, tiers, swimlanes), \
layers and any implied sequence or hierarchy.
5. Inferred behaviour: briefly state what the system appears to do end to end. Mark all \
inference clearly as inferred.

Use plain text with short labelled sections and bullet lines. Be complete rather than \
concise; include details even if they look minor. If something is illegible, say \
"[illegible]" rather than guessing."""

MECHANICAL_PROMPT = """You are analysing a mechanical engineering drawing or schematic. \
Describe it exhaustively as a single detailed text report so that it can be understood \
without seeing the image.

Cover, in this order:
1. Drawing type and title block: title, drawing number, revision, scale, units, material \
and any notes written in the title block.
2. Views: orthographic, isometric, section, detail and auxiliary views, their labels and \
what each view shows.
3. Components and features: every part, assembly, hole, slot, boss, rib, thread, gear, \
bearing, fastener, spring or weld shown. Give labels, part numbers and apparent function.
4. Dimensions and annotations: transcribe every dimension, tolerance, datum, surface \
finish, thread callout, weld symbol, material specification and note verbatim.
5. Relationships and motion: assembly hierarchy, mating parts, fits, clearances, \
fasteners, exploded-view order, axes of rotation and any implied motion.
6. Inferred behaviour: briefly state what the part or assembly appears to do. Mark all \
inference clearly as inferred.

Use plain text with short labelled sections and bullet lines. Be complete rather than \
concise. If something is illegible, say "[illegible]" rather than guessing."""

CIVIL_PROMPT = """You are analysing a civil engineering or architectural drawing. Describe \
it exhaustively as a single detailed text report so that it can be understood without \
seeing the image.

Cover, in this order:
1. Drawing type and title block: plan, elevation, section, site plan, structural detail or \
utility drawing; title, sheet number, revision, scale, north arrow and units.
2. Elements: every site boundary, road, path, building, foundation, column, beam, slab, \
wall, stair, ramp, drain, pipe, utility line, landscape feature and annotation shown. Give \
labels and apparent function.
3. Dimensions and levels: transcribe every dimension, grid line, level, elevation, slope, \
section marker, material callout, reinforcement note and specification verbatim.
4. Structure and relationships: grouping, levels, grid references, connection details, \
cross-references to other sheets and any sequence or hierarchy.
5. Inferred behaviour: briefly state what the site or structure appears to be for. Mark all \
inference clearly as inferred.

Use plain text with short labelled sections and bullet lines. Be complete rather than \
concise. If something is illegible, say "[illegible]" rather than guessing."""

CATEGORY_PROMPTS: dict[str, str] = {
    "mechanical": MECHANICAL_PROMPT,
    "civil": CIVIL_PROMPT,
    "computer_flow": PROMPT,
}


def prompt_for_category(category: str | None) -> str:
    """Return the vision prompt best suited to ``category``."""
    return CATEGORY_PROMPTS.get(category or "", PROMPT)


def encode_image(path: Path) -> str:
    """Return a ``data:`` URL containing the base64-encoded image."""
    raw = path.read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"{path.name} is {len(raw) / 1_048_576:.1f} MB, above the 20 MB API limit"
        )
    mime = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def describe_image(
    client: Any,
    path: Path,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompt: str | None = None,
) -> str:
    """Ask the model to describe one diagram and return its text."""
    data_url = encode_image(path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("model returned an empty response")
    return content.strip()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file so a crash can never truncate the results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def find_images(directory: Path) -> list[Path]:
    """Return the supported image files in ``directory``, sorted by name."""
    if not directory.is_dir():
        raise NotADirectoryError(f"folder not found: {directory}")
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_existing_descriptions(path: Path) -> dict[str, str]:
    """Load an existing descriptions file, keeping only string entries."""
    if not path.is_file():
        return {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        print(
            f"warning: could not read {path} ({exc}); starting fresh",
            file=sys.stderr,
        )
        return {}
    if not isinstance(existing, dict):
        print(
            f"warning: {path} does not contain a JSON object; starting fresh",
            file=sys.stderr,
        )
        return {}
    return {k: v for k, v in existing.items() if isinstance(v, str)}


def describe_diagrams(
    *,
    diagrams_dir: Path = DEFAULT_DIAGRAMS_DIR,
    output: Path = DEFAULT_DESCRIPTIONS_FILE,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    resume: bool = False,
    fail_fast: bool = False,
    categories: dict[str, str] | None = None,
    client: Any | None = None,
) -> int:
    """Describe every image in ``diagrams_dir`` and write the results to ``output``.

    Returns a process exit code: 0 on success, 1 when some images failed, and 2
    when the input or configuration is invalid.
    """
    if not diagrams_dir.is_dir():
        print(f"error: folder not found: {diagrams_dir}", file=sys.stderr)
        return 2

    images = find_images(diagrams_dir)
    if not images:
        print(
            f"No .png/.jpg/.jpeg files found in {diagrams_dir}",
            file=sys.stderr,
        )
        return 1

    results: dict[str, str] = {}
    if resume:
        results = load_existing_descriptions(output)
        if results:
            print(f"Resuming: {len(results)} description(s) already in {output}")

    if client is None:
        require_api_key()
        client = OpenAI(max_retries=5, timeout=120.0)

    total = len(images)
    failures: list[tuple[str, str]] = []
    started = time.monotonic()

    for index, path in enumerate(images, start=1):
        name = path.name
        if name in results:
            print(f"[{index}/{total}] {name}: already described, skipping")
            continue

        print(f"[{index}/{total}] {name}: describing...", end="", flush=True)
        t0 = time.monotonic()
        category = (categories or {}).get(name)
        try:
            results[name] = describe_image(
                client,
                path,
                model,
                max_tokens,
                prompt=prompt_for_category(category),
            )
        except (OpenAIError, ValueError, OSError) as exc:
            print(f" failed ({exc})")
            failures.append((name, str(exc)))
            if fail_fast:
                break
        else:
            print(
                f" done in {time.monotonic() - t0:.1f}s "
                f"({len(results[name]):,} chars)"
            )
        finally:
            # Save after every image so partial progress survives an interruption.
            write_json_atomic(output, results)

    elapsed = time.monotonic() - started
    print(f"\nWrote {len(results)} description(s) to {output} in {elapsed:.1f}s")

    if failures:
        print(f"{len(failures)} image(s) failed:", file=sys.stderr)
        for name, err in failures:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1

    return 0

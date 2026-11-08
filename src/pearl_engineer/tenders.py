"""Tender crawler for gravity / water filter defence tenders.

This module finds customers asking for tenders on **gravity filters** and
**water filters** across Indian defence projects. It crawls multiple tender
portals (GeM, CPPP/eProcure, DRDO, MES, Ministry of Defence and public
aggregators), matches listings against gravity-filter / water-filter
keywords, de-duplicates them into a local JSON store, and produces a gist
summary (who is buying, what, how much, and by when).

Design notes
------------
* No hard dependency on BeautifulSoup or network access: the default fetcher
  uses the standard library (``urllib``) and falls back to ``httpx`` when it
  is installed. HTML is parsed with the standard ``html.parser`` plus targeted
  regexes, so the feature — and its tests — run fully offline.
* A ``fetcher`` callable can be injected for tests or for portals that need
  authenticated / headless-browser access. Real portals such as GeM and CPPP
  frequently render with JavaScript or require rate-limited access; the
  crawler therefore degrades gracefully (per-source errors are recorded in the
  :class:`CrawlReport` rather than failing the whole run) and also offers a
  DuckDuckGo HTML search fallback that works without keys.
* Records are stored in ``data/tenders.json`` (override with
  ``PEARL_TENDERS_FILE``). Re-crawling is idempotent: records are keyed by a
  stable hash of URL + tender number + title.

Install the optional crawl extra for ``httpx`` + ``beautifulsoup4``::

    pip install "pearl-engineer[crawl]"

Example::

    from pathlib import Path
    from pearl_engineer.tenders import TenderCrawler, TenderStore, summarize_tenders

    store = TenderStore(Path("data/tenders.json"))
    report = TenderCrawler(store).crawl()
    print(report)
    gist = summarize_tenders(store.list())
    print(gist["headline"])
"""

from __future__ import annotations

import hashlib
import html as _html_module
import json
import logging
import re
import threading
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords and defence context
# ---------------------------------------------------------------------------

#: Search queries used when the caller does not supply their own.
DEFAULT_QUERIES: tuple[str, ...] = (
    "gravity filter",
    "gravity water filter",
    "water filter defence",
    "gravity filter tender",
    "slow sand gravity filter",
)

#: Exact phrases that indicate a gravity / water filter requirement.
GRAVITY_TERMS: tuple[str, ...] = (
    "gravity filter",
    "gravity fed filter",
    "gravity-fed filter",
    "gravity water filter",
    "slow sand filter",
    "rapid gravity filter",
    "gravity filtration",
)

#: Broader water-filter terms (kept when paired with a gravity/defence signal).
WATER_TERMS: tuple[str, ...] = (
    "water filter",
    "water filtration",
    "water purification",
    "drinking water filter",
    "water treatment filter",
)

#: Signals that a tender belongs to the defence / military ecosystem.
DEFENCE_TERMS: tuple[str, ...] = (
    "defence",
    "defense",
    "army",
    "navy",
    "air force",
    "airforce",
    "drdo",
    "ordnance",
    "military",
    "ministry of defence",
    "military engineer",
    "border security",
    "bsf",
    "camp",
    "battalion",
    "regiment",
    " cantonment",
    "mes ",
)

#: Minimum relevance score for a listing to be kept (0..1).
RELEVANCE_THRESHOLD = 0.30

USER_AGENT = "PearlEngineer-TenderCrawler/1.1 (+https://github.com/)"


def snippet_only(text: str) -> str:
    """Return the first sentence of ``text`` to avoid neighbouring listings."""
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0][:240] if text else ""


_DATE_RE = re.compile(
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    re.IGNORECASE,
)
_TENDER_NO_RE = re.compile(
    r"\b(?:GEM/[A-Z0-9/-]+|GEMC/[A-Z0-9/-]+|[A-Z]{2,}/[A-Z0-9/-]+/\d{4}|"
    r"Tender\s*No\.?\s*[:#]?\s*[A-Z0-9/-]{4,40}|"
    r"Bid\s*No\.?\s*[:#]?\s*[A-Z0-9/-]{4,40})\b",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)
_ANCHOR_RE = re.compile(
    r'<a\b[^>]*\bhref\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_VALUE_RE = re.compile(
    r"\b(?:Rs\.?|INR|₹)\s*([\d,.]+)\s*(lakh|crore|cr|lacs|thousand)?\b",
    re.IGNORECASE,
)
_CAPACITY_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:LPH|KLH|litres?\s*per\s*hour|L/hr|m3/hr|units?)\b",
    re.IGNORECASE,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm_text(value: str | None) -> str:
    if not value:
        return ""
    return _html_module.unescape(_TAG_RE.sub(" ", value))


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def score_relevance(title: str, snippet: str = "", buyer: str = "") -> float:
    """Score how strongly a listing matches gravity-filter defence tenders.

    Pure, deterministic and offline — no ML model required. The score is in
    ``[0, 1]``; listings at or above :data:`RELEVANCE_THRESHOLD` are kept.
    """
    text = f"{title} {snippet} {buyer}".lower()
    score = 0.0
    # Exact gravity-filter phrase is the strongest signal.
    if any(term in text for term in GRAVITY_TERMS):
        score += 0.55
    elif "gravity" in text and "filter" in text:
        score += 0.45
    elif "water filter" in text:
        score += 0.30
    elif any(term in text for term in WATER_TERMS):
        score += 0.18
    if any(term in text for term in DEFENCE_TERMS):
        score += 0.25
    if "tender" in text or "bid" in text or "rfp" in text or "rfq" in text:
        score += 0.10
    if "filter" in text and "water" in text:
        score += 0.10
    return min(1.0, score)


def extract_dates(text: str) -> list[str]:
    """Extract ISO dates (YYYY-MM-DD) from free text, in order found."""
    found: list[str] = []
    for match in _DATE_RE.finditer(text):
        raw = match.group(1).strip()
        iso = _parse_date(raw)
        if iso and iso not in found:
            found.append(iso)
    return found


def _parse_date(raw: str) -> str | None:
    for fmt in (
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d-%m-%y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%d %b %Y",
        "%d %B %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
    ):
        try:
            parsed = datetime.strptime(raw.strip(), fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return None


def extract_tender_number(text: str) -> str | None:
    match = _TENDER_NO_RE.search(text)
    return _collapse(match.group(0)) if match else None


def extract_buyer(text: str, fallback: str | None = None) -> str | None:
    """Best-effort buyer extraction from a listing title/snippet."""
    # Check in specificity order: full organisation names first, then short
    # aliases (e.g. a title that just says "Army" or "Naval").
    candidates: tuple[tuple[str, str], ...] = (
        ("MINISTRY OF DEFENCE", "Ministry of Defence"),
        ("MOD", "Ministry of Defence"),
        ("INDIAN ARMY", "Indian Army"),
        ("NAVY", "Indian Navy"),
        ("INDIAN NAVY", "Indian Navy"),
        ("INDIAN AIR FORCE", "Indian Air Force"),
        ("AIR FORCE", "Indian Air Force"),
        ("DEFENCE RESEARCH AND DEVELOPMENT ORGANISATION", "DRDO"),
        ("DRDO", "DRDO"),
        ("MILITARY ENGINEER SERVICES", "Military Engineer Services"),
        ("MES", "Military Engineer Services"),
        ("BORDER SECURITY FORCE", "Border Security Force"),
        ("BSF", "Border Security Force"),
        ("GOVERNMENT E-MARKETPLACE", "GeM"),
        ("GEM", "GeM"),
        ("CPWD", "CPWD"),
        ("CENTRAL PUBLIC WORKS DEPARTMENT", "CPWD"),
        ("ORDNANCE FACTORY", "Ordnance Factory"),
        ("ARMY", "Indian Army"),
    )
    upper = f" {_TENDER_NO_RE.sub(' ', text).upper()} "
    for needle, label in candidates:
        if needle in upper:
            # Avoid "MES"/"GEM"/"MOD" matching inside other words.
            if len(needle) <= 4:
                if not re.search(rf"\b{re.escape(needle)}\b", upper):
                    continue
            return label
    labelled = re.search(
        r"(?:Buyer|Department|Organisation|Organization|Purchaser|Authority)\s*[:\-]\s*([A-Za-z][A-Za-z0-9 &().,'-]{2,80})",
        text,
        re.IGNORECASE,
    )
    if labelled:
        return _collapse(labelled.group(1))
    return fallback


def tender_id_for(*, url: str, tender_no: str | None, title: str) -> str:
    """Stable id for a tender: hash of normalised URL + number + title."""
    key = "|".join(
        [
            urllib.parse.urldefrag(url or "")[0].strip().lower(),
            (tender_no or "").strip().lower(),
            re.sub(r"\s+", " ", (title or "").strip().lower()),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenderSource:
    """One portal the crawler can search.

    ``search_template`` contains a ``{query}`` placeholder that is replaced
    with the URL-encoded query. Sources whose template is a directory listing
    (no ``{query}``) are fetched once and parsed for tender links.
    """

    id: str
    name: str
    search_template: str
    kind: str = "portal"  # portal | aggregator | search
    base_url: str = ""
    description: str = ""

    def search_url(self, query: str) -> str:
        encoded = urllib.parse.quote_plus(query)
        return self.search_template.replace("{query}", encoded)


#: Built-in defence tender portals. URLs are public search endpoints; the
#: crawler treats every fetch as best-effort and records failures per source.
DEFAULT_SOURCES: tuple[TenderSource, ...] = (
    TenderSource(
        id="gem",
        name="GeM (Government e-Marketplace)",
        search_template="https://gem.gov.in/search?query={query}",
        kind="portal",
        base_url="https://gem.gov.in",
        description="Central government e-marketplace for defence buys, incl. gravity filters.",
    ),
    TenderSource(
        id="cppp",
        name="CPPP / eProcure (GoI)",
        search_template=("https://cppp.nic.in/cppp/search?module=view&title={query}"),
        kind="portal",
        base_url="https://cppp.nic.in",
        description="Central Public Procurement Portal — MoD, MES, DRDO, CPWD notices.",
    ),
    TenderSource(
        id="eprocure",
        name="eProcure (GoI tender portal)",
        search_template=(
            "https://eprocure.gov.in/eprocure/app?component=${'$'}Direct"
            "&page=searchN.inc&service=direct&query={query}"
        ),
        kind="portal",
        base_url="https://eprocure.gov.in",
        description="National e-procurement listings for defence and infrastructure.",
    ),
    TenderSource(
        id="drdo",
        name="DRDO Tenders",
        search_template="https://www.drdo.gov.in/tenders?search={query}",
        kind="portal",
        base_url="https://www.drdo.gov.in",
        description="Defence Research & Development Organisation tenders.",
    ),
    TenderSource(
        id="mod",
        name="Ministry of Defence Tenders",
        search_template="https://www.mod.gov.in/tenders?search={query}",
        kind="portal",
        base_url="https://www.mod.gov.in",
        description="MoD procurement notices, incl. army/navy/air-force campuses.",
    ),
    TenderSource(
        id="mes",
        name="Military Engineer Services (MES)",
        search_template="https://mes.gov.in/tenders?search={query}",
        kind="portal",
        base_url="https://mes.gov.in",
        description="MES works & equipment tenders for cantonments and stations.",
    ),
    TenderSource(
        id="tender247",
        name="Tender247 (aggregator)",
        search_template="https://www.tender247.com/search?query={query}",
        kind="aggregator",
        base_url="https://www.tender247.com",
        description="Aggregated Indian tenders with defence and water-filter filters.",
    ),
    TenderSource(
        id="tenderdetail",
        name="TenderDetail (aggregator)",
        search_template="https://www.tendersdetail.com/search?query={query}",
        kind="aggregator",
        base_url="https://www.tendersdetail.com",
        description="Aggregated defence tenders including water treatment.",
    ),
    TenderSource(
        id="duckduckgo",
        name="DuckDuckGo search fallback",
        search_template=(
            "https://html.duckduckgo.com/html/?q={query}+gravity+filter+"
            "defence+tender+site%3Agem.gov.in+OR+site%3Acppp.nic.in+OR+site%3Adrdo.gov.in"
        ),
        kind="search",
        base_url="https://html.duckduckgo.com",
        description="Keyless public search fallback for defence gravity-filter bids.",
    ),
)


def get_sources(ids: Iterable[str] | None = None) -> list[TenderSource]:
    """Return sources filtered by ``id`` (all when ``ids`` is empty/None)."""
    if not ids:
        return list(DEFAULT_SOURCES)
    wanted = {str(item).strip().lower() for item in ids if str(item).strip()}
    if not wanted:
        return list(DEFAULT_SOURCES)
    selected = [s for s in DEFAULT_SOURCES if s.id.lower() in wanted]
    return selected or list(DEFAULT_SOURCES)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenderRecord:
    """One tender / advertisement for a gravity or water filter."""

    id: str
    title: str
    buyer: str | None = None
    tender_no: str | None = None
    source_id: str = ""
    source_name: str = ""
    url: str = ""
    publish_date: str | None = None
    closing_date: str | None = None
    location: str | None = None
    value: str | None = None
    quantity: str | None = None
    description: str = ""
    snippet: str = ""
    keywords: tuple[str, ...] = ()
    discovered_at: str = ""
    relevance: float = 0.0
    status: str = "unknown"  # open | closed | unknown

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["keywords"] = list(self.keywords)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TenderRecord:
        return cls(
            id=str(data.get("id") or ""),
            title=str(data.get("title") or "Untitled tender"),
            buyer=data.get("buyer"),
            tender_no=data.get("tender_no"),
            source_id=str(data.get("source_id") or ""),
            source_name=str(data.get("source_name") or ""),
            url=str(data.get("url") or ""),
            publish_date=data.get("publish_date"),
            closing_date=data.get("closing_date"),
            location=data.get("location"),
            value=data.get("value"),
            quantity=data.get("quantity"),
            description=str(data.get("description") or ""),
            snippet=str(data.get("snippet") or ""),
            keywords=tuple(data.get("keywords") or ()),
            discovered_at=str(data.get("discovered_at") or ""),
            relevance=float(data.get("relevance") or 0.0),
            status=str(data.get("status") or "unknown"),
        )


@dataclass
class CrawlReport:
    """Outcome of one crawler run."""

    fetched: int = 0
    found: int = 0
    new: int = 0
    updated: int = 0
    errors: list[str] = field(default_factory=list)
    sources_tried: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    demo: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TenderStore:
    """JSON-backed store for crawled tenders (thread-safe, idempotent)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[str, TenderRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.warning("could not load tenders from %s: %s", self.path, exc)
            return
        items: list[Any]
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("tenders"), list):
            items = payload["tenders"]
        else:
            logger.warning("tenders file %s has unexpected shape; ignoring", self.path)
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            record = TenderRecord.from_dict(item)
            if record.id:
                self._records[record.id] = record

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tenders": [self._records[key].to_dict() for key in sorted(self._records)],
            "updated_at": _now_iso(),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(self.path)

    def upsert_many(self, records: Iterable[TenderRecord]) -> tuple[int, int]:
        """Insert or update records; returns ``(new, updated)`` counts."""
        new = updated = 0
        with self._lock:
            for record in records:
                if record.id in self._records:
                    updated += 1
                else:
                    new += 1
                self._records[record.id] = record
            if new or updated:
                self._save_locked()
        return new, updated

    def list(
        self,
        *,
        source: str | None = None,
        buyer: str | None = None,
        q: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[TenderRecord]:
        """List tenders sorted by closing date (soonest first), then newest."""
        with self._lock:
            records = list(self._records.values())
        if source:
            records = [r for r in records if r.source_id.lower() == source.lower()]
        if buyer:
            needle = buyer.lower()
            records = [r for r in records if r.buyer and needle in r.buyer.lower()]
        if q:
            needle = q.lower()
            records = [
                r
                for r in records
                if needle in f"{r.title} {r.snippet} {r.description}".lower()
            ]
        if status:
            records = [r for r in records if r.status == status.lower()]
        records.sort(
            key=lambda r: (r.closing_date or "9999", r.discovered_at), reverse=False
        )
        return records[: max(1, limit)] if limit else records

    def get(self, tender_id: str) -> TenderRecord | None:
        with self._lock:
            return self._records.get(tender_id)

    def count(self) -> int:
        with self._lock:
            return len(self._records)


# ---------------------------------------------------------------------------
# HTML parsing (stdlib only, no BeautifulSoup required)
# ---------------------------------------------------------------------------


def parse_tender_links(
    html_text: str,
    *,
    base_url: str,
    source: TenderSource,
    query: str = "",
) -> list[TenderRecord]:
    """Parse tender links from a portal search-results page.

    The parser looks for anchors whose text mentions a filter/tender signal,
    then enriches each candidate with buyer, tender number, dates and a
    relevance score. Listings below :data:`RELEVANCE_THRESHOLD` are dropped.
    """
    candidates: list[TenderRecord] = []
    seen_urls: set[str] = set()
    page_text = _collapse(_norm_text(html_text))
    discovered_at = _now_iso()
    all_matches = list(_ANCHOR_RE.finditer(html_text or ""))
    for index, match in enumerate(all_matches):
        raw_href = (match.group(1) or "").strip()
        if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        anchor_inner = _collapse(_norm_text(match.group(2)))
        if len(anchor_inner) < 6:
            continue
        absolute = urllib.parse.urljoin(base_url, _html_module.unescape(raw_href))
        if absolute in seen_urls:
            continue
        # Context: a narrow window bounded by neighbouring anchors so values
        # from other listings do not bleed into this record.
        lower_bound = all_matches[index - 1].end() if index > 0 else 0
        upper_bound = (
            all_matches[index + 1].start()
            if index + 1 < len(all_matches)
            else len(html_text)
        )
        start = max(lower_bound, match.start() - 200)
        end = min(upper_bound, match.end() + 250)
        context = _collapse(_norm_text(html_text[start:end]))
        combined = f"{anchor_inner} {context} {query}"
        # Score the anchor text itself (context only adds a small defence
        # bonus) so neighbouring listings cannot inflate unrelated links.
        relevance = score_relevance(anchor_inner, "", "")
        lowered_context = context.lower()
        if any(term in lowered_context for term in DEFENCE_TERMS):
            relevance = min(1.0, relevance + 0.10)
        # Keep anchors that look like a filter/tender requirement even when
        # the surrounding window is noisy.
        lowered = anchor_inner.lower()
        if relevance < RELEVANCE_THRESHOLD and not (
            "filter" in lowered
            and ("tender" in lowered or "bid" in lowered or "water" in lowered)
        ):
            continue
        if relevance < RELEVANCE_THRESHOLD:
            continue

        seen_urls.add(absolute)
        title = anchor_inner[:300]
        snippet = context[:500]
        tender_no = extract_tender_number(f"{title} {snippet}")
        buyer = extract_buyer(title) or extract_buyer(snippet, fallback=source.name)
        dates = extract_dates(snippet)
        publish_date = dates[0] if dates else None
        closing_date = dates[-1] if len(dates) > 1 else (dates[0] if dates else None)
        value_match = _VALUE_RE.search(snippet)
        value = _collapse(value_match.group(0)) if value_match else None
        capacity_match = _CAPACITY_RE.search(snippet)
        quantity = _collapse(capacity_match.group(0)) if capacity_match else None
        keywords = tuple(
            term for term in GRAVITY_TERMS + WATER_TERMS if term in combined.lower()
        )
        status = _status_for(closing_date)
        candidates.append(
            TenderRecord(
                id=tender_id_for(url=absolute, tender_no=tender_no, title=title),
                title=title,
                buyer=buyer,
                tender_no=tender_no,
                source_id=source.id,
                source_name=source.name,
                url=absolute,
                publish_date=publish_date,
                closing_date=closing_date,
                value=value,
                quantity=quantity,
                description=snippet[:2000],
                snippet=snippet[:500],
                keywords=keywords,
                discovered_at=discovered_at,
                relevance=round(relevance, 3),
                status=status,
            )
        )
    # Fallback: whole-page scan for tender numbers near a filter mention,
    # so JS-rendered listings that still expose plain text are not missed.
    if not candidates and page_text:
        lowered_page = page_text.lower()
        if "gravity" in lowered_page and "filter" in lowered_page:
            for hit in re.finditer(
                r".{0,80}gravity.{0,80}filter.{0,120}", page_text, re.IGNORECASE
            ):
                snippet = _collapse(hit.group(0))
                title = (
                    _collapse(
                        re.split(r"(?<=Tender|Bid|RFP)", snippet, flags=re.IGNORECASE)[
                            0
                        ]
                    )[:300]
                    or "Gravity filter tender"
                )
                tender_no = extract_tender_number(snippet)
                relevance = score_relevance(title, snippet, "")
                if relevance < RELEVANCE_THRESHOLD:
                    continue
                dates = extract_dates(snippet)
                candidates.append(
                    TenderRecord(
                        id=tender_id_for(
                            url=f"{base_url}#gravity-{len(candidates)}",
                            tender_no=tender_no,
                            title=title,
                        ),
                        title=title,
                        buyer=extract_buyer(snippet, fallback=source.name),
                        tender_no=tender_no,
                        source_id=source.id,
                        source_name=source.name,
                        url=base_url,
                        publish_date=dates[0] if dates else None,
                        closing_date=dates[-1] if dates else None,
                        description=snippet[:2000],
                        snippet=snippet[:500],
                        keywords=("gravity filter",),
                        discovered_at=discovered_at,
                        relevance=round(relevance, 3),
                        status=_status_for(dates[-1] if dates else None),
                    )
                )
    return candidates


def _status_for(closing_date: str | None) -> str:
    if not closing_date:
        return "unknown"
    today = datetime.now(timezone.utc).date().isoformat()
    return "open" if closing_date >= today else "closed"


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

Fetcher = Callable[[str], str]


def default_fetcher(
    url: str, *, timeout: float = 12.0, max_bytes: int = 2_000_000
) -> str:
    """Fetch ``url`` as text (httpx when available, else urllib)."""
    try:  # pragma: no cover - depends on optional install
        import httpx  # type: ignore

        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,*/*",
            },
        )
        response.raise_for_status()
        return response.text[:max_bytes]
    except ImportError:
        pass
    request = urllib.request.Request(  # noqa: S310 - fixed internal portal URLs
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:  # noqa: S310
        raw = handle.read(max_bytes + 1)
        charset = handle.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------


class TenderCrawler:
    """Multi-portal crawler for gravity / water filter tenders."""

    def __init__(
        self,
        store: TenderStore,
        *,
        sources: list[TenderSource] | None = None,
        fetcher: Fetcher | None = None,
        timeout: float = 12.0,
    ) -> None:
        self.store = store
        self.sources = sources or list(DEFAULT_SOURCES)
        self._fetcher = fetcher
        self.timeout = timeout

    def _fetch(self, url: str) -> str:
        if self._fetcher is not None:
            return self._fetcher(url)
        return default_fetcher(url, timeout=self.timeout)

    def crawl(
        self,
        *,
        queries: list[str] | None = None,
        source_ids: list[str] | None = None,
        demo: bool = False,
    ) -> CrawlReport:
        """Crawl every configured portal × query and store new tenders.

        ``demo=True`` seeds the store with realistic sample tenders (no
        network) — useful for local development, UI previews and tests.
        """
        active_queries = [
            q.strip() for q in (queries or list(DEFAULT_QUERIES)) if q and q.strip()
        ]
        if not active_queries:
            active_queries = list(DEFAULT_QUERIES)
        active_sources = get_sources(source_ids) if source_ids else list(self.sources)

        report = CrawlReport(
            sources_tried=[s.id for s in active_sources],
            queries=active_queries,
            demo=demo,
        )

        if demo:
            records = sample_tenders()
            new, updated = self.store.upsert_many(records)
            report.found = len(records)
            report.new = new
            report.updated = updated
            logger.info(
                "seeded %d sample tenders (%d new, %d updated)",
                len(records),
                new,
                updated,
            )
            return report

        all_records: list[TenderRecord] = []
        for source in active_sources:
            for query in active_queries:
                url = source.search_url(query)
                try:
                    html_text = self._fetch(url)
                    report.fetched += 1
                except Exception as exc:  # per-source failure must not sink the run
                    message = f"{source.id} ({query!r}): fetch failed: {exc}"
                    logger.warning(message)
                    report.errors.append(message)
                    continue
                try:
                    records = parse_tender_links(
                        html_text, base_url=url, source=source, query=query
                    )
                except Exception as exc:  # pragma: no cover - defensive parser guard
                    message = f"{source.id} ({query!r}): parse failed: {exc}"
                    logger.warning(message)
                    report.errors.append(message)
                    continue
                all_records.extend(records)

        # De-duplicate within this run before writing.
        deduped: dict[str, TenderRecord] = {}
        for record in all_records:
            existing = deduped.get(record.id)
            if existing is None or record.relevance > existing.relevance:
                deduped[record.id] = record
        report.found = len(deduped)
        new, updated = self.store.upsert_many(deduped.values())
        report.new = new
        report.updated = updated
        return report


# ---------------------------------------------------------------------------
# Gist summary
# ---------------------------------------------------------------------------


def summarize_tenders(
    records: Iterable[TenderRecord],
    *,
    today: str | None = None,
) -> dict[str, Any]:
    """Build the deterministic gist summary for a set of tenders.

    No LLM is required: the summary aggregates counts by source and buyer,
    sorts upcoming deadlines, and extracts the most common requirement terms
    so the UI (and CLI) can show a trustworthy "gist" even offline.
    """
    items = list(records)
    today_iso = today or datetime.now(timezone.utc).date().isoformat()
    by_source = Counter(r.source_name or r.source_id or "Unknown" for r in items)
    by_buyer = Counter(r.buyer or "Unspecified" for r in items)
    keyword_counts: Counter[str] = Counter()
    for record in items:
        for keyword in record.keywords:
            keyword_counts[keyword] += 1
        lowered = f"{record.title} {record.snippet}".lower()
        for term in GRAVITY_TERMS + WATER_TERMS:
            if term in lowered and term not in record.keywords:
                keyword_counts[term] += 1

    def _due_key(record: TenderRecord) -> tuple[str, str]:
        return (record.closing_date or "9999-12-31", record.title.lower())

    upcoming = sorted(
        [r for r in items if r.closing_date and r.closing_date >= today_iso],
        key=_due_key,
    )
    closed = [r for r in items if r.closing_date and r.closing_date < today_iso]
    unknown = [r for r in items if not r.closing_date]
    top_buyers = [name for name, _ in by_buyer.most_common(5)]
    top_keywords = [term for term, _ in keyword_counts.most_common(8)]

    if not items:
        headline = "No gravity-filter tenders found yet — run a crawl."
        gist = (
            "No tenders match gravity-filter / water-filter criteria in the "
            "current store. Use the crawl action to search GeM, CPPP/eProcure, "
            "DRDO, MES, MoD and aggregator portals."
        )
    else:
        portal_word = f"{len(by_source)} portal(s)"
        upcoming_note = (
            f"{len(upcoming)} close on a known future date"
            if upcoming
            else "closing dates are still being collected"
        )
        buyer_note = (
            f"mainly {', '.join(top_buyers[:3])}"
            if top_buyers
            else "buyers not yet identified"
        )
        headline = (
            f"Found {len(items)} gravity / water filter tender(s) across {portal_word} "
            f"— {upcoming_note}."
        )
        gist = (
            f"{len(items)} tender(s) were found across {len(by_source)} portal(s) "
            f"({', '.join(name for name, _ in by_source.most_common(4))}). "
            f"Buyers are {buyer_note}. "
            f"{len(upcoming)} tender(s) have an upcoming closing date"
            f"{' (soonest: ' + upcoming[0].closing_date + ' — ' + upcoming[0].title[:80] + ')' if upcoming else ''}. "
            f"{len(closed)} tender(s) have already closed and {len(unknown)} have no parsed date. "
            f"Most common requirement terms: {', '.join(top_keywords) if top_keywords else 'gravity filter, water filter'}."
        )

    return {
        "total": len(items),
        "open": len(upcoming),
        "closed": len(closed),
        "unknown_dates": len(unknown),
        "by_source": dict(by_source),
        "by_buyer": dict(by_buyer),
        "keywords": dict(keyword_counts),
        "top_keywords": top_keywords,
        "top_buyers": top_buyers,
        "upcoming": [r.to_dict() for r in upcoming[:10]],
        "headline": headline,
        "gist": gist,
        "generated_at": _now_iso(),
    }


def summarize_tenders_with_llm(
    records: Iterable[TenderRecord],
    *,
    client: Any | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Deterministic gist plus an optional LLM paragraph.

    When ``client`` is None (or the call fails) the deterministic gist is
    returned unchanged, so the API always works without an API key.
    """
    items = list(records)
    base = summarize_tenders(items)
    base["llm_gist"] = None
    if client is None or not items:
        return base
    lines = [
        f"- {r.title} | buyer={r.buyer or 'n/a'} | closing={r.closing_date or 'n/a'} | {r.url}"
        for r in items[:25]
    ]
    prompt = (
        "Summarise these gravity-filter / water-filter defence tenders into a "
        "concise sales gist (3-5 bullets): who is buying, what capacity/quantity, "
        "deadlines, and the most urgent opportunities. Do not invent numbers.\n\n"
        + "\n".join(lines)
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You summarise Indian defence tenders."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            base["llm_gist"] = text
    except Exception as exc:  # pragma: no cover - network/LLM failure path
        logger.warning("LLM gist failed: %s", exc)
    return base


# ---------------------------------------------------------------------------
# Sample data (offline demo / tests)
# ---------------------------------------------------------------------------


def sample_tenders() -> list[TenderRecord]:
    """Realistic sample tenders for offline demo and tests (not live data)."""
    now = _now_iso()
    raw: list[dict[str, Any]] = [
        {
            "title": "Supply of gravity filter water purification units for Army cantonment",
            "buyer": "Indian Army",
            "tender_no": "GEM/2026/B/7401234",
            "source_id": "gem",
            "source_name": "GeM (Government e-Marketplace)",
            "url": "https://gem.gov.in/bid/7401234",
            "publish_date": "2026-10-20",
            "closing_date": "2026-11-25",
            "location": "Pune, Maharashtra",
            "value": "Rs. 48 lakh",
            "quantity": "5000 LPH",
            "snippet": "Gravity filter units, 5000 LPH capacity, for drinking water supply at cantonment.",
            "description": "Qty 12 units of gravity-fed water filter, 5000 LPH, inclusive of installation.",
        },
        {
            "title": "Tender for slow sand gravity filter plant — DRDO research station",
            "buyer": "DRDO",
            "tender_no": "DRDO/2026/WS/0098",
            "source_id": "drdo",
            "source_name": "DRDO Tenders",
            "url": "https://www.drdo.gov.in/tenders/0098",
            "publish_date": "2026-10-28",
            "closing_date": "2026-12-05",
            "location": "Hyderabad, Telangana",
            "value": "Rs. 1.2 crore",
            "quantity": "20 KLH",
            "snippet": "Slow sand gravity filtration plant for laboratory and drinking water.",
            "description": "Design, supply and commissioning of 20 KLH slow sand gravity filter plant.",
        },
        {
            "title": "MES — Rapid gravity water filter for border outpost, BSF camp",
            "buyer": "Military Engineer Services",
            "tender_no": "MES/2026/B/2210",
            "source_id": "mes",
            "source_name": "Military Engineer Services (MES)",
            "url": "https://mes.gov.in/tenders/2210",
            "publish_date": "2026-10-15",
            "closing_date": "2026-11-30",
            "location": "Jaisalmer, Rajasthan",
            "value": "Rs. 32 lakh",
            "quantity": "3000 LPH",
            "snippet": "Rapid gravity filter for drinking water at border outpost.",
            "description": "Supply of 3000 LPH rapid gravity water filter units with spares.",
        },
        {
            "title": "CPPP — Water filter tender: gravity filtration for naval station",
            "buyer": "Indian Navy",
            "tender_no": "NWS/2026/WF/114",
            "source_id": "cppp",
            "source_name": "CPPP / eProcure (GoI)",
            "url": "https://cppp.nic.in/cppp/notice/114",
            "publish_date": "2026-11-01",
            "closing_date": "2026-12-18",
            "location": "Visakhapatnam, Andhra Pradesh",
            "value": "Rs. 75 lakh",
            "quantity": "10000 LPH",
            "snippet": "Gravity filtration plant, 10000 LPH, for naval station drinking water.",
            "description": "Turnkey gravity water filter plant 10000 LPH for naval station.",
        },
        {
            "title": "GeM Bid: Gravity filter media and spares for defence camp maintenance",
            "buyer": "GeM",
            "tender_no": "GEM/2026/B/7519902",
            "source_id": "gem",
            "source_name": "GeM (Government e-Marketplace)",
            "url": "https://gem.gov.in/bid/7519902",
            "publish_date": "2026-11-02",
            "closing_date": "2026-12-10",
            "location": "Bengaluru, Karnataka",
            "value": "Rs. 18 lakh",
            "quantity": "500 units",
            "snippet": "Gravity filter media, cartridges and spares for ongoing defence camp maintenance.",
            "description": "Annual rate contract for gravity filter media and spares.",
        },
        {
            "title": "Air Force station — Water filter AMC and gravity filter replacement",
            "buyer": "Indian Air Force",
            "tender_no": "IAF/2026/AMC/077",
            "source_id": "mod",
            "source_name": "Ministry of Defence Tenders",
            "url": "https://www.mod.gov.in/tenders/077",
            "publish_date": "2026-09-30",
            "closing_date": "2026-10-30",
            "location": "Pathankot, Punjab",
            "value": "Rs. 22 lakh",
            "quantity": "AMC",
            "snippet": "Comprehensive AMC for water filters including gravity filter replacement.",
            "description": "AMC for installed water filters and gravity filter overhauling.",
        },
    ]
    records: list[TenderRecord] = []
    for item in raw:
        relevance = score_relevance(item["title"], item["snippet"], item["buyer"])
        keywords = tuple(
            term
            for term in GRAVITY_TERMS + WATER_TERMS
            if term in f"{item['title']} {item['snippet']}".lower()
        ) or ("gravity filter",)
        records.append(
            TenderRecord(
                id=tender_id_for(
                    url=item["url"], tender_no=item["tender_no"], title=item["title"]
                ),
                title=item["title"],
                buyer=item["buyer"],
                tender_no=item["tender_no"],
                source_id=item["source_id"],
                source_name=item["source_name"],
                url=item["url"],
                publish_date=item.get("publish_date"),
                closing_date=item.get("closing_date"),
                location=item.get("location"),
                value=item.get("value"),
                quantity=item.get("quantity"),
                description=item.get("description", ""),
                snippet=item.get("snippet", ""),
                keywords=keywords,
                discovered_at=now,
                relevance=round(relevance, 3),
                status=_status_for(item.get("closing_date")),
            )
        )
    return records


__all__ = [
    "DEFAULT_QUERIES",
    "DEFAULT_SOURCES",
    "DEFENCE_TERMS",
    "GRAVITY_TERMS",
    "WATER_TERMS",
    "CrawlReport",
    "TenderCrawler",
    "TenderRecord",
    "TenderSource",
    "TenderStore",
    "default_fetcher",
    "extract_buyer",
    "extract_dates",
    "extract_tender_number",
    "get_sources",
    "parse_tender_links",
    "sample_tenders",
    "score_relevance",
    "summarize_tenders",
    "summarize_tenders_with_llm",
    "tender_id_for",
]

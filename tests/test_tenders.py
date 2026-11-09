"""Tests for the gravity-filter tender crawler (fully offline)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pearl_engineer.config import CacheSettings, SearchSettings
from pearl_engineer.search import create_app
from pearl_engineer.tenders import (
    DEFAULT_SOURCES,
    TenderCrawler,
    TenderStore,
    get_sources,
    parse_tender_links,
    sample_tenders,
    score_relevance,
    summarize_tenders,
)

HTML_FIXTURE = """
<html><body>
<a href="/tenders/gravity-filter-5000lph">Supply of gravity filter water
purification units for Army cantonment — Tender No. GEM/2026/B/7401234</a>
<p>Army welfare, closing 25-11-2026, value Rs. 48 lakh</p>
<a href="/tenders/office-chairs">Office chairs and tables for HQ</a>
<a href="/tenders/water-filter-drdo">Slow sand gravity filter plant — DRDO
research station, closing 05-12-2026</a>
</body></html>
"""


def test_score_relevance_prefers_gravity_defence() -> None:
    strong = score_relevance(
        "gravity filter tender for Army", "water filtration", "Indian Army"
    )
    weak = score_relevance("office furniture", "chairs for HQ", "Office")
    assert strong > 0.7
    assert weak < 0.3


def test_parse_tender_links_extracts_gravity_tenders() -> None:
    source = DEFAULT_SOURCES[0]
    records = parse_tender_links(
        HTML_FIXTURE, base_url="https://gem.gov.in/search", source=source
    )
    assert len(records) == 2
    titles = " ".join(record.title.lower() for record in records)
    assert "gravity filter" in titles
    assert "office chairs" not in titles
    for record in records:
        assert record.source_id == source.id
        assert record.url.startswith("https://gem.gov.in/")
        assert record.relevance >= 0.3
    first = [r for r in records if "7401234" in (r.url + r.title)][0]
    assert first.tender_no is not None
    assert "GEM/2026/B/7401234" in first.tender_no
    assert first.buyer == "Indian Army"
    assert first.closing_date == "2026-11-25"


def test_get_sources_filters_by_id() -> None:
    assert [s.id for s in get_sources(["gem", "drdo"])] == ["gem", "drdo"]
    assert get_sources(None) == list(DEFAULT_SOURCES)


def test_store_upsert_is_idempotent(tmp_path: Path) -> None:
    store = TenderStore(tmp_path / "tenders.json")
    first = sample_tenders()
    new, updated = store.upsert_many(first)
    assert (new, updated) == (len(first), 0)
    new2, updated2 = store.upsert_many(first)
    assert (new2, updated2) == (0, len(first))
    assert store.count() == len(first)
    assert store.get(first[0].id) is not None


def test_summarize_builds_gist() -> None:
    records = sample_tenders()
    gist = summarize_tenders(records, today="2026-11-08")
    assert gist["total"] == len(records)
    assert gist["headline"].startswith("Found")
    assert "gravity" in gist["gist"].lower()
    assert gist["open"] >= 1
    assert gist["by_source"]
    assert gist["top_keywords"]
    assert gist["upcoming"]  # at least one upcoming deadline vs 2026-11-08
    assert gist["upcoming"][0]["closing_date"] >= "2026-11-08"


def test_crawler_with_fake_fetcher(tmp_path: Path) -> None:
    store = TenderStore(tmp_path / "tenders.json")
    crawler = TenderCrawler(
        store,
        sources=[DEFAULT_SOURCES[0]],
        fetcher=lambda url: HTML_FIXTURE,
    )
    report = crawler.crawl(queries=["gravity filter"])
    assert report.fetched == 1
    assert report.found == 2
    assert report.new == 2
    assert store.count() == 2


def test_crawler_demo_seeds_without_network(tmp_path: Path) -> None:
    store = TenderStore(tmp_path / "tenders.json")
    crawler = TenderCrawler(
        store, fetcher=lambda url: (_ for _ in ()).throw(AssertionError("no fetch"))
    )
    report = crawler.crawl(demo=True)
    assert report.demo is True
    assert report.new == len(sample_tenders())
    assert store.count() == len(sample_tenders())


def test_crawler_records_per_source_errors(tmp_path: Path) -> None:
    def failing(url: str) -> str:
        raise RuntimeError("portal blocked")

    store = TenderStore(tmp_path / "tenders.json")
    crawler = TenderCrawler(store, sources=[DEFAULT_SOURCES[0]], fetcher=failing)
    report = crawler.crawl(queries=["gravity filter"])
    assert report.found == 0
    assert report.errors  # error is recorded, not raised
    assert store.count() == 0


def _make_client(tmp_path: Path) -> tuple[TestClient, TenderStore]:
    store = TenderStore(tmp_path / "tenders.json")
    settings = SearchSettings(
        persist_dir=tmp_path / "chroma",
        collection="diagrams",
        embedding_model="test-embedding-model",
        cache=CacheSettings(enabled=False),
        admin_token="",
        diagrams_dir=tmp_path / "diagrams",
        descriptions_file=tmp_path / "descriptions.json",
        categories_file=tmp_path / "categories.json",
        vision_model="test-vision-model",
        tenders_file=tmp_path / "tenders.json",
    )

    class _FakeCollection:
        def count(self) -> int:
            return 0

        def query(self, **kwargs):  # pragma: no cover - unused
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    from types import SimpleNamespace

    fake_openai = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda *, model, input: SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
            )
        )
    )
    app = create_app(
        settings,
        collection=_FakeCollection(),
        openai_client=fake_openai,
        tender_store=store,
    )
    return TestClient(app), store


def test_tender_api_list_and_summary(tmp_path: Path) -> None:
    client, store = _make_client(tmp_path)

    with client:
        empty = client.get("/tenders")
        assert empty.status_code == 200
        assert empty.json() == []

        crawled = client.post("/tenders/crawl", json={"demo": True})
        assert crawled.status_code == 200
        body = crawled.json()
        assert body["new"] == len(sample_tenders())
        assert body["summary"]["total"] == len(sample_tenders())

        listed = client.get("/tenders?source=gem")
        assert listed.status_code == 200
        assert all(item["source_id"] == "gem" for item in listed.json())
        assert listed.json()

        summary = client.get("/tenders/summary")
        assert summary.status_code == 200
        assert summary.json()["total"] == len(sample_tenders())
        assert "gravity" in summary.json()["gist"].lower()

        sources = client.get("/tenders/sources")
        assert sources.status_code == 200
        assert {item["id"] for item in sources.json()} >= {"gem", "cppp", "drdo"}

        one = client.get(f"/tenders/{listed.json()[0]['id']}")
        assert one.status_code == 200
        assert one.json()["title"]

        missing = client.get("/tenders/does-not-exist")
        assert missing.status_code == 404

    assert store.count() == len(sample_tenders())


def test_tender_api_crawl_requires_admin_for_live(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    # Rebuild with an admin token set.
    from pearl_engineer.config import CacheSettings as _CS

    store = TenderStore(tmp_path / "tenders2.json")
    settings = SearchSettings(
        persist_dir=tmp_path / "chroma",
        collection="diagrams",
        embedding_model="test-embedding-model",
        cache=_CS(enabled=False),
        admin_token="secret",
        diagrams_dir=tmp_path / "diagrams",
        descriptions_file=tmp_path / "descriptions.json",
        categories_file=tmp_path / "categories.json",
        vision_model="test-vision-model",
        tenders_file=tmp_path / "tenders2.json",
    )
    from types import SimpleNamespace

    fake_openai = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda *, model, input: SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
            )
        )
    )

    class _FakeCollection:
        def count(self) -> int:
            return 0

    app = create_app(
        settings,
        collection=_FakeCollection(),
        openai_client=fake_openai,
        tender_store=store,
    )
    live_client = TestClient(app)
    with live_client:
        unauthorized = live_client.post("/tenders/crawl", json={})
        demo_allowed = live_client.post("/tenders/crawl", json={"demo": True})
    assert unauthorized.status_code == 401
    assert demo_allowed.status_code == 200


def test_health_includes_tenders(tmp_path: Path) -> None:
    client, store = _make_client(tmp_path)
    store.upsert_many(sample_tenders()[:2])
    with client:
        health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["tenders"] == 2

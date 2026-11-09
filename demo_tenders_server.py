"""Demo server for Tender Ads screenshot — no ChromaDB / API key needed."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("OPENAI_API_KEY", "sk-demo-screenshot")

from pearl_engineer.config import CacheSettings, SearchSettings  # noqa: E402
from pearl_engineer.search import create_app  # noqa: E402
from pearl_engineer.tenders import TenderStore  # noqa: E402

import uvicorn  # noqa: E402


class FakeCollection:
    def count(self) -> int:
        return 0

    def query(self, **kwargs):  # pragma: no cover
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def main() -> None:
    tenders_file = ROOT / "data" / "tenders.json"
    settings = SearchSettings(
        persist_dir=ROOT / "data" / "chroma_db",
        collection="diagrams",
        embedding_model="test-embedding-model",
        cache=CacheSettings(enabled=False),
        admin_token="",
        diagrams_dir=ROOT / "diagrams",
        descriptions_file=ROOT / "data" / "descriptions.json",
        categories_file=ROOT / "data" / "categories.json",
        vision_model="gpt-4o-mini",
        max_upload_bytes=20 * 1024 * 1024,
        tenders_file=tenders_file,
    )
    fake_openai = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda *, model, input: SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
            )
        )
    )
    app = create_app(
        settings,
        collection=FakeCollection(),
        openai_client=fake_openai,
        tender_store=TenderStore(tenders_file),
    )
    print(f"Demo server ready: http://127.0.0.1:8001/ (tenders: {tenders_file})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")


if __name__ == "__main__":
    main()

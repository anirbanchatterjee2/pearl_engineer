# Pearl Engineer

Semantic search and question answering over architecture diagrams.

The core pipeline has three stages:

1. **Describe** every diagram image with an OpenAI vision model and save the text.
2. **Index** those descriptions as embeddings in a local ChromaDB collection.
3. **Search** the collection through a small FastAPI service.

Diagrams can be ingested in bulk from the CLI or one at a time through the
upload API and bundled web UI. Every record can carry a **category**
(`mechanical`, `civil` or `computer_flow`) and an optional title, so search can
be filtered and results can be labelled.

A production retrieval layer sits on top of those stages:

- a **two-tier semantic cache** (exact query matching plus embedding similarity),
- a shared **`SearchService`** used by HTTP, LangChain, LangGraph and the CLI,
- optional **LangChain** retriever and tool adapters,
- an optional **LangGraph** retrieval/generation workflow with query refinement,
- cache metrics, health checks and an admin-protected cache invalidation endpoint.

## Features

| Capability | What it gives you |
| --- | --- |
| Semantic cache | Exact and cosine-similarity hits, TTL, LRU eviction, namespaces and hit-rate metrics |
| Shared service | One retrieval implementation for HTTP, agents, chains and scripts |
| Categories | Mechanical, civil and computer-flow diagrams with metadata filtering |
| Upload & web UI | Admin-protected upload/describe/index endpoint and a bundled browser UI |
| LangChain | `BaseRetriever` and `BaseTool` adapters with sync and async support |
| LangGraph | Inspectable retrieve → refine → generate graph, optional checkpointer and ReAct agent |
| Observability | `/health`, `/cache/stats`, cache headers and structured counters |
| Safety | Configurable admin token for cache invalidation and uploads, bounded caches, upload validation and clear error mapping |

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                 SearchService                │
                    │  embed → semantic cache → ChromaDB → parse   │
                    └───────────────┬──────────────┬───────────────┘
                                    │              │
             ┌──────────────────────┘              └──────────────────────┐
             │                                                            │
     ┌───────▼────────┐              ┌──────────────────┐          ┌──────▼───────┐
     │  FastAPI       │              │  LangChain       │          │  LangGraph   │
     │  POST /search  │              │  retriever/tool  │          │  retrieve    │
     │  /diagrams     │              │  chains/agents   │          │  refine      │
     │  /categories   │              │                  │          │  generate    │
     │  /health       │              │                  │          │              │
     │  /cache/*      │              │                  │          │              │
     └───────┬────────┘              └──────────────────┘          └──────┬───────┘
             │                                                            │
             └───────────────────────────┬────────────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  SemanticCache      │
                              │  exact + cosine     │
                              │  TTL + LRU + stats  │
                              └─────────────────────┘
```

## Project structure

```
pearl-engineer/
├── diagrams/                       # input images (gitignored)
│   └── .gitkeep
├── data/                           # generated artefacts (gitignored)
│   ├── descriptions.json           # created by `pearl describe`
│   ├── categories.json             # optional filename → category map
│   └── chroma_db/                  # created by `pearl index`
├── src/
│   └── pearl_engineer/
│       ├── __init__.py             # package version
│       ├── __main__.py             # `python -m pearl_engineer`
│       ├── cli.py                  # argparse commands + console scripts
│       ├── config.py               # env vars, paths, cache settings
│       ├── categories.py           # diagram categories and aliases
│       ├── describe.py             # vision-model descriptions
│       ├── ingest.py               # upload validation, describe + index
│       ├── index.py                # embeddings + ChromaDB upserts
│       ├── semantic_cache.py       # enterprise semantic cache
│       ├── service.py              # shared retrieval service
│       ├── search.py               # FastAPI app factory and models
│       ├── integrations/
│       │   ├── __init__.py
│       │   ├── langchain.py        # retriever + tool adapters
│       │   └── langgraph.py        # retrieval/generation graph
│       ├── static/                 # bundled web UI
│       │   ├── index.html
│       │   ├── app.js
│       │   └── styles.css
│       └── py.typed                # PEP 561 marker
├── tests/
│   ├── test_describe.py
│   ├── test_index.py
│   ├── test_search.py
│   ├── test_service.py
│   ├── test_semantic_cache.py
│   ├── test_upload.py
│   ├── test_langchain_integration.py
│   └── test_langgraph_integration.py
├── .env.example
├── .gitignore
├── pyproject.toml                  # package metadata + dependencies
├── requirements.txt                # core runtime deps
├── requirements-dev.txt            # core + test/lint deps
├── requirements-langchain.txt      # optional LangChain deps
├── requirements-langgraph.txt      # optional LangGraph deps
├── requirements-enterprise.txt     # full enterprise stack
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Option A (recommended): install the package and dev tools
pip install -e ".[dev]"

# Option B: plain requirements files
pip install -r requirements-dev.txt
pip install -e . --no-deps

# Optional agent integrations
pip install -e ".[langchain]"
pip install -e ".[langgraph]"

# Or everything at once
pip install -e ".[dev,enterprise]"

Copy-Item .env.example .env
# then edit .env and set OPENAI_API_KEY
```

`pyproject.toml` is the source of truth for package metadata and dependencies.
The `requirements*.txt` files mirror those lists for workflows that use plain
`pip install -r ...`; keep them in sync if you change a version.

The package loads `.env` from the working directory automatically. You can also
set the key directly:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

## Usage

Put `.png`, `.jpg`, or `.jpeg` diagrams in `diagrams/`, then run the pipeline:

```powershell
pearl describe          # write data/descriptions.json
pearl index             # embed and upsert into data/chroma_db
pearl serve             # http://127.0.0.1:8000
pearl ask "how does billing consume order events?"
```

The same commands are available through the module:

```powershell
python -m pearl_engineer describe
python -m pearl_engineer index
python -m pearl_engineer serve
python -m pearl_engineer ask "..."
```

Individual console scripts are provided too: `pearl-describe`, `pearl-index`,
`pearl-serve`, and `pearl-ask`.

### Describe options

```powershell
pearl describe --diagrams-dir ./diagrams --output ./data/descriptions.json
pearl describe --categories ./data/categories.json
pearl describe --resume        # keep existing entries and skip those images
pearl describe --fail-fast     # stop at the first API error
pearl describe --model gpt-4o --max-tokens 6000
```

Results are saved after every image, so an interrupted run keeps its progress.

### Diagram categories

Categories are optional metadata stored on every indexed record:

| ID | Label | Typical content |
| --- | --- | --- |
| `mechanical` | Mechanical | Parts, assemblies, tolerances, materials |
| `civil` | Civil Engineering | Structural, site, utility and elevation drawings |
| `computer_flow` | Computer Flow | Software architecture, data flow and process diagrams |

Bulk ingestion reads an optional `data/categories.json` map from filename to
category (aliases such as `flowchart` or `structural` are normalised):

```json
{
  "orders.png": "computer_flow",
  "bracket.png": "mechanical",
  "site-plan.png": "civil"
}
```

The category also selects the vision prompt used by `pearl describe`. Files
without an entry are indexed as `uncategorized`.

### Index options

```powershell
pearl index --dry-run          # show what would be indexed, no API calls
pearl index --reset            # drop the collection and rebuild it
pearl index --categories ./data/categories.json
pearl index --batch-size 32 --max-chars 30000
pearl index --collection diagrams --persist-dir ./data/chroma_db
```

Re-running is idempotent: record IDs are image filenames, so changed
descriptions are overwritten rather than duplicated.

### Ask with LangGraph

`pearl ask` runs the retrieval/generation graph from the command line:

```powershell
pearl ask "which services publish order events to Kafka?"
pearl ask "what database does billing use?" --n-results 8
pearl ask "summarise the payment flow" --no-llm     # retrieval only
pearl ask "..." --json                              # full graph state
pearl ask "..." --chat-model gpt-4o --model text-embedding-3-large
```

It requires the optional LangGraph extra. `--no-llm` still requires LangGraph
because the workflow itself is the graph; it simply skips the chat model and
returns a deterministic digest of the retrieved descriptions.

## Semantic cache

The cache is implemented in `pearl_engineer.semantic_cache` and is wired into
`SearchService`, so HTTP requests, LangChain tools, LangGraph nodes and
`pearl ask` all share the same behaviour.

### How lookup works

1. **Exact tier** — the query is normalised (case, leading/trailing whitespace
   and repeated whitespace are ignored). An exact hit returns immediately
   without calling the embedding API.
2. **Semantic tier** — on an exact miss the query is embedded and compared with
   cached query vectors using cosine similarity. A score at or above
   `PEARL_CACHE_SIMILARITY_THRESHOLD` returns the cached result.
3. **Miss** — the service queries ChromaDB, stores the result under a namespace
   and returns it.

Namespaces include the collection, embedding model, requested result count and
any category filter, so changing the model, asking for a different number of
results or switching category cannot return an incompatible payload.

### Production behaviour

- Thread-safe for FastAPI's worker thread pool.
- TTL expiry with lazy pruning on reads, writes and stats.
- LRU eviction when `PEARL_CACHE_MAX_ENTRIES` is reached.
- Embedding happens outside the cache lock, so a slow provider does not block
  other readers or writers.
- Hit, miss, store, eviction and expiration counters are exposed through
  `GET /cache/stats` and `GET /health`.
- Cached payloads are immutable `SearchResult` tuples, safe to share between
  concurrent requests.

### Cache configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `PEARL_CACHE_ENABLED` | `true` | Enable the semantic cache |
| `PEARL_CACHE_SIMILARITY_THRESHOLD` | `0.92` | Minimum cosine similarity for a semantic hit |
| `PEARL_CACHE_TTL_SECONDS` | `3600` | Entry lifetime; `0` disables expiry |
| `PEARL_CACHE_MAX_ENTRIES` | `1024` | Maximum entries before LRU eviction |
| `PEARL_ADMIN_TOKEN` | empty | Required in `X-Admin-Token` for `POST /cache/clear` |

Tuning guidance:

- **Higher threshold** (`0.95`–`0.99`) — fewer, safer hits; best when answers
  must be exact.
- **Lower threshold** (`0.85`–`0.92`) — more reuse for paraphrased questions;
  monitor the cache hit rate and sample hits for quality.
- **Short TTL** — use when the index changes frequently.
- **Long TTL** — use for stable architecture documentation.

The cache is process-local. A single uvicorn worker gets a single cache; with
multiple workers each process has its own. For a shared multi-host cache, put a
vector-capable cache such as Redis with RediSearch in front of the retrieval
service or replace `SemanticCache` behind the same `SearchService` interface.

### Cache CLI flags

`pearl serve` accepts cache overrides that map onto the environment:

```powershell
pearl serve --cache
pearl serve --no-cache
pearl serve --cache-threshold 0.95 --cache-ttl 900 --cache-size 4096
```

## Search API

```powershell
# POST /search (optionally filtered by category)
curl.exe -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query": "how does the order service talk to billing?", "n_results": 5, "category": "computer_flow"}'

# GET /categories
curl.exe http://127.0.0.1:8000/categories

# GET /diagrams?category=computer_flow
curl.exe "http://127.0.0.1:8000/diagrams?category=computer_flow&limit=50"

# POST /diagrams — upload, describe and index one image
curl.exe -X POST http://127.0.0.1:8000/diagrams `
  -H "X-Admin-Token: $env:PEARL_ADMIN_TOKEN" `
  -F "file=@./diagrams/orders.png" `
  -F "category=computer_flow" `
  -F "title=Orders"

# GET /diagrams/{filename} and /diagrams/{filename}/download
curl.exe http://127.0.0.1:8000/diagrams/orders.png

# GET /health
curl.exe http://127.0.0.1:8000/health

# GET /cache/stats
curl.exe http://127.0.0.1:8000/cache/stats

# POST /cache/clear (all namespaces)
curl.exe -X POST http://127.0.0.1:8000/cache/clear

# POST /cache/clear?namespace=diagrams%7Ctext-embedding-3-small%7Cn%3D5
curl.exe -X POST "http://127.0.0.1:8000/cache/clear?namespace=..." `
  -H "X-Admin-Token: $env:PEARL_ADMIN_TOKEN"
```

The bundled web UI is served at `/` and talks to the same endpoints. Uploads
are limited to PNG/JPEG files and `PEARL_MAX_UPLOAD_BYTES` (20 MB by default).
When `PEARL_ADMIN_TOKEN` is set, both uploads and cache invalidation require the
`X-Admin-Token` header.

Example response:

```json
{
  "query": "how does the order service talk to billing?",
  "results": [
    {
      "description": "...",
      "image_path": "O:\\My Projects\\pearl-engineer\\diagrams\\orders.png",
      "image_file": "orders.png",
      "distance": 0.1823,
      "similarity": 0.8177,
      "category": "computer_flow",
      "category_label": "Computer Flow",
      "title": "Orders",
      "image_url": "/diagrams/orders.png",
      "download_url": "/diagrams/orders.png/download"
    }
  ],
  "cache": {
    "enabled": true,
    "hit": false,
    "kind": null,
    "similarity": null,
    "age_seconds": null
  }
}
```

Every `/search` response also carries cache diagnostics in headers:

| Header | Meaning |
| --- | --- |
| `X-Cache` | `HIT` or `MISS` |
| `X-Cache-Kind` | `exact` or `semantic` on a hit |
| `X-Cache-Similarity` | Similarity score for a semantic hit |
| `X-Cache-Age` | Age of the cached entry in seconds |

`/health` reports `status`, `collection`, `records` and a `cache` object with
`entries`, `namespaces`, `capacity`, `hits`, `exact_hits`, `semantic_hits`,
`misses`, `lookups`, `hit_rate`, `stores`, `evictions`, `expirations`,
`ttl_seconds` and `similarity_threshold`.

Error mapping:

- `503` — collection missing or not built.
- `502` — embedding or vision provider failure.
- `500` — ChromaDB query failure.
- `401` — upload or cache clear attempted without the configured admin token.
- `413` — upload exceeds the configured size limit.
- `415` — upload is not a PNG/JPEG image.
- `422` — empty upload, unknown category or invalid request body.

## LangChain integration

Install the optional extra:

```powershell
pip install -e ".[langchain]"
```

Build a `SearchService` once and reuse it:

```python
import chromadb
from openai import OpenAI

from pearl_engineer.semantic_cache import SemanticCache
from pearl_engineer.service import SearchService

collection = chromadb.PersistentClient(path="data/chroma_db").get_collection("diagrams")
service = SearchService(
    collection=collection,
    embedding_client=OpenAI(),
    embedding_model="text-embedding-3-small",
    collection_name="diagrams",
    cache=SemanticCache(similarity_threshold=0.92),
)
```

### Retriever

```python
from pearl_engineer.integrations.langchain import diagram_retriever

retriever = diagram_retriever(service, n_results=5)
documents = retriever.invoke("which service publishes order events?")

for document in documents:
    print(document.metadata["image_file"], document.page_content[:120])
```

`DiagramRetriever` returns LangChain `Document` objects whose metadata contains
`image_file`, `image_path`, `category`, `title`, `distance` and `similarity`
plus any other ChromaDB metadata. Async invocation is supported through
`ainvoke`.

### Tool

```python
from pearl_engineer.integrations.langchain import diagram_search_tool

tool = diagram_search_tool(service, n_results=5)
answer_context = tool.invoke({"query": "billing database", "n_results": 3})
```

`DiagramSearchTool` returns compact text for an LLM, truncating long
descriptions at `max_description_chars` (default `4000`). Both `invoke` and
`ainvoke` are supported, and tool calls are traced through LangChain callbacks.

### Agent usage

The tool can be passed to any LangChain agent or chain:

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an architecture assistant. Cite diagram filenames."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, [tool], prompt)
executor = AgentExecutor(agent=agent, tools=[tool])
print(executor.invoke({"input": "how does billing consume order events?"})["output"])
```

## LangGraph integration

Install the optional extra:

```powershell
pip install -e ".[langgraph]"
```

### Explicit workflow

```python
from langchain_openai import ChatOpenAI

from pearl_engineer.integrations.langgraph import build_diagram_graph

graph = build_diagram_graph(
    service,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    n_results=5,
    max_refinements=1,
)

result = graph.invoke(
    {
        "query": "how does billing consume order events?",
        "original_query": "how does billing consume order events?",
        "refinements": 0,
    }
)

print(result["answer"])
print(result["sources"])
```

The graph contains three nodes:

| Node | Responsibility |
| --- | --- |
| `retrieve` | Calls the LangChain retriever and records sources |
| `refine` | Expands the query deterministically when nothing was found |
| `generate` | Produces a grounded answer, or a deterministic digest without an LLM |

`max_refinements` controls how many query expansions are attempted. Pass a
custom `query_refiner(original_query, attempt) -> str` to use an LLM or a domain
thesaurus. Pass a LangGraph `checkpointer` to persist conversation state, and
`thread_id` to `ask()` for resumable threads.

### Offline mode

```python
graph = build_diagram_graph(service, llm=None)
result = graph.invoke(
    {"query": "payment flow", "original_query": "payment flow", "refinements": 0}
)
print(result["answer"])  # deterministic digest of retrieved descriptions
```

This is useful for smoke tests, evaluation harnesses and environments without a
chat model.

### ReAct agent

```python
from pearl_engineer.integrations.langgraph import build_react_agent

agent = build_react_agent(
    service=service,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    n_results=5,
)
result = agent.invoke(
    {"messages": [("user", "which services publish order events?")]},
    config={"configurable": {"thread_id": "demo"}},
)
print(result["messages"][-1].content)
```

`build_react_agent` uses `langgraph.prebuilt.create_react_agent` when the
installed LangGraph version provides it; otherwise use the explicit graph above.

## Configuration

All settings are environment variables; the defaults are shown below.

| Variable | Default | Used by |
| --- | --- | --- |
| `OPENAI_API_KEY` | — (required) | describe, index, serve, ask |
| `PEARL_DIAGRAMS_DIR` | `diagrams` | describe, index, serve (upload target) |
| `PEARL_DATA_DIR` | `data` | base for generated files |
| `PEARL_DESCRIPTIONS_FILE` | `data/descriptions.json` | describe, index |
| `PEARL_CATEGORIES_FILE` | `data/categories.json` | describe, index, serve |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | index, serve, ask |
| `CHROMA_COLLECTION` | `diagrams` | index, serve, ask |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | index, serve, ask |
| `PEARL_VISION_MODEL` | `gpt-4o-mini` | describe |
| `PEARL_CHAT_MODEL` | `gpt-4o-mini` | ask, LangGraph |
| `PEARL_CACHE_ENABLED` | `true` | serve, ask |
| `PEARL_CACHE_SIMILARITY_THRESHOLD` | `0.92` | serve, ask |
| `PEARL_CACHE_TTL_SECONDS` | `3600` | serve, ask |
| `PEARL_CACHE_MAX_ENTRIES` | `1024` | serve, ask |
| `PEARL_ADMIN_TOKEN` | empty | serve (`/cache/clear`, `POST /diagrams`) |
| `PEARL_MAX_UPLOAD_BYTES` | `20971520` (20 MB) | serve (upload limit) |

## Development

```powershell
pytest                  # unit tests; no API key or network required
ruff check .
ruff format .
```

Tests inject fake OpenAI and ChromaDB clients, so they run fully offline. The
LangChain and LangGraph tests use `pytest.importorskip`; install the matching
extra to run them:

```powershell
pip install -e ".[dev,enterprise]"
pytest
```

## Gravity-filter tender crawler

The crawler finds customers asking for **gravity filters** and **water filters**
in Indian defence projects. It searches multiple portals (GeM,
CPPP/eProcure, DRDO, MES, Ministry of Defence and public aggregators),
filters listings by gravity-filter relevance, stores unique tenders in
`data/tenders.json`, and produces a **gist summary** (who is buying, what,
and by when).

```powershell
pip install -e ".[crawl]"              # httpx + beautifulsoup4 (optional)
pearl crawl-tenders                     # live crawl + gist summary (offline-safe)
pearl crawl-tenders --demo              # seed realistic sample tenders, no network
pearl crawl-tenders --list              # print stored tenders + gist without crawling
pearl crawl-tenders --sources gem drdo  # crawl selected portals only
pearl crawl-tenders --json              # machine-readable report + tenders
```

HTTP API (bundled UI tab **Tender Ads**, shortcut `T`):

```powershell
# GET /tenders?source=gem&q=DRDO  — list advertisements, soonest closing first
# GET /tenders/summary             — gist headline + counts by portal/buyer
# GET /tenders/sources             — crawlable portals
# POST /tenders/crawl {"demo": true}  — seed demo data (no token needed)
# POST /tenders/crawl {}              — live crawl (needs X-Admin-Token when set)
```

Live crawls are best-effort: each portal failure is recorded in the response
and never blocks other portals. GeM/CPPP often require JavaScript or rate
limits; the DuckDuckGo fallback and demo data keep the UI usable offline.
Override the store with `PEARL_TENDERS_FILE`.

## Migrating from the old scripts

| Old command | New command |
| --- | --- |
| `python describe_diagrams.py` | `pearl describe` |
| `python index_diagrams.py` | `pearl index` |
| `uvicorn search_api:app --reload` | `pearl serve --reload` |
| `python search_api.py` | `pearl serve` |
| — | `pearl ask "..."` |

Default output paths moved from the repository root into `data/`. Override them
with `--output` / `--persist-dir` or the environment variables above if you want
to keep the old layout.

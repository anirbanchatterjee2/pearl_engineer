# Pearl Engineer

Semantic search over architecture diagrams. The pipeline has three stages:

1. **Describe** every diagram image with an OpenAI vision model and save the text.
2. **Index** those descriptions as embeddings in a local ChromaDB collection.
3. **Search** the collection through a small FastAPI service.

## Project structure

```
pearl-engineer/
├── diagrams/                     # input images (gitignored)
│   └── .gitkeep
├── data/                         # generated artefacts (gitignored)
│   ├── descriptions.json         # created by `pearl describe`
│   └── chroma_db/                # created by `pearl index`
├── src/
│   └── pearl_engineer/
│       ├── __init__.py           # package version
│       ├── __main__.py           # `python -m pearl_engineer`
│       ├── cli.py                # argparse commands + console scripts
│       ├── config.py             # env vars, paths, defaults
│       ├── describe.py           # vision-model descriptions
│       ├── index.py              # embeddings + ChromaDB upserts
│       └── search.py             # FastAPI app factory and models
├── tests/
│   ├── test_describe.py
│   ├── test_index.py
│   └── test_search.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

Copy-Item .env.example .env
# then edit .env and set OPENAI_API_KEY
```

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
```

The same commands are available through the module:

```powershell
python -m pearl_engineer describe
python -m pearl_engineer index
python -m pearl_engineer serve
```

Individual console scripts are provided too: `pearl-describe`, `pearl-index`,
and `pearl-serve`.

### Describe options

```powershell
pearl describe --diagrams-dir ./diagrams --output ./data/descriptions.json
pearl describe --resume        # keep existing entries and skip those images
pearl describe --fail-fast     # stop at the first API error
pearl describe --model gpt-4o --max-tokens 6000
```

Results are saved after every image, so an interrupted run keeps its progress.

### Index options

```powershell
pearl index --dry-run          # show what would be indexed, no API calls
pearl index --reset            # drop the collection and rebuild it
pearl index --batch-size 32 --max-chars 30000
pearl index --collection diagrams --persist-dir ./data/chroma_db
```

Re-running is idempotent: record IDs are image filenames, so changed
descriptions are overwritten rather than duplicated.

### Search API

```powershell
# POST /search
curl.exe -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d '{"query": "how does the order service talk to billing?", "n_results": 5}'

# GET /health
curl.exe http://127.0.0.1:8000/health
```

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
      "similarity": 0.8177
    }
  ]
}
```

If the collection has not been built, `/search` returns `503`; if the query
cannot be embedded, it returns `502`.

## Configuration

All settings are environment variables; the defaults are shown below.

| Variable | Default | Used by |
| --- | --- | --- |
| `OPENAI_API_KEY` | — (required) | describe, index, serve |
| `PEARL_DIAGRAMS_DIR` | `diagrams` | describe, index |
| `PEARL_DATA_DIR` | `data` | base for generated files |
| `PEARL_DESCRIPTIONS_FILE` | `data/descriptions.json` | describe, index |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | index, serve |
| `CHROMA_COLLECTION` | `diagrams` | index, serve |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | index, serve |
| `PEARL_VISION_MODEL` | `gpt-4o-mini` | describe |

## Development

```powershell
pytest                  # unit tests; no API key or network required
ruff check .
ruff format .
```

Tests inject fake OpenAI and ChromaDB clients, so they run fully offline.

## Migrating from the old scripts

| Old command | New command |
| --- | --- |
| `python describe_diagrams.py` | `pearl describe` |
| `python index_diagrams.py` | `pearl index` |
| `uvicorn search_api:app --reload` | `pearl serve --reload` |
| `python search_api.py` | `pearl serve` |

Default output paths moved from the repository root into `data/`. Override them
with `--output` / `--persist-dir` or the environment variables above if you want
to keep the old layout.

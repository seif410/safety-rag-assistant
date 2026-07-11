# safety-rag-assistant

RAG-powered Q&A over OSHA safety regulations and incident reports — ask natural-language
questions and get source-cited answers across two document types (formal regulations and
field incident reports).

## What makes this different

Most RAG demos stop at vector similarity. This one adds the pieces production retrieval
actually needs:

- **Metadata-filtered retrieval** — chunks carry `doc_type` (`regulation` / `incident_report`),
  `filename`, and `page`, so the retrieval tool can scope a search to regulations or incident
  reports instead of searching one undifferentiated blob.
- **Reranking** — retrieve 6 candidates by vector similarity, then rerank with Cohere and keep
  the top 4 before generation.
- **Dual document types** — formal PDFs (OSHA regulations) and unstructured Markdown field
  reports, ingested through one pipeline.
- **Offline evaluation** — a hand-written Q&A set scored with hit-rate@k, MRR, and an
  LLM-as-judge faithfulness metric (see [Evaluation](#evaluation)).
- **Self-hosted, API-first** — Qdrant runs locally via Docker (no vector-DB API key), FastAPI
  serves the endpoints, and conversational memory is kept per session.

## Architecture

```
                        ┌─────────────────────────────┐
  PDFs (OSHA regs)  ──►  │ Ingestion                   │
  Markdown (reports) ──► │ PyMuPDF / MD reader         │
                        │ → chunk (1200 / 200 overlap) │
                        │ → NVIDIA embeddings          │
                        └──────────────┬──────────────┘
                                       ▼
                              ┌──────────────────┐
                              │ Qdrant (Docker)  │  collection: safety-docs
                              └────────┬─────────┘
                                       ▼
             ┌────────────────────────────────────────────┐
   query ──► │ Retrieval (k=6, optional doc_type filter)   │
             │   → Cohere rerank (top_n=4)                 │
             └───────────────────────┬────────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ Agent (Gemini 2.5 Flash)        │  + per-session memory
                    │   retrieve_safety_docs tool     │
                    └───────────────┬─────────────────┘
                                    ▼
                     answer + cited sources (file, page, type)
```

## Tech stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12+ |
| API | FastAPI / Uvicorn | 0.139.0 / 0.49.0 |
| Orchestration | LangChain / langchain-core | 1.3.11 / 1.4.8 |
| Vector store | Qdrant (server via Docker) / langchain-qdrant / qdrant-client | latest / 1.1.0 / 1.18.0 |
| Embeddings | NVIDIA `llama-nemotron-embed-1b-v2` (langchain-nvidia-ai-endpoints) | 1.4.3 |
| Reranker | Cohere `rerank-v4.0-pro` (langchain-cohere) | 0.6.0 |
| Chat model | Google `gemini-2.5-flash` (langchain-google-genai) | 4.2.6 |
| PDF parsing | PyMuPDF | 1.28.0 |
| Config / schemas | pydantic / pydantic-settings | 2.13.4 / 2.14.2 |
| Packaging | uv | — |

The chat model id and provider are configurable via `CHAT_MODEL` / `CHAT_MODEL_PROVIDER`
(defaults `gemini-2.5-flash` / `google_genai`); the agent authenticates with `GOOGLE_API_KEY`.

## Quick start

Requires Docker and API keys for NVIDIA (embeddings), Cohere (rerank), and Google (chat).

```bash
# 1. Configure secrets  (Windows cmd: copy .env.example .env)
cp .env.example .env
# edit .env and fill in NVIDIA_API_KEY, COHERE_API_KEY, and GOOGLE_API_KEY

# 2. Start Qdrant + the API
docker compose up -d --build

# 3. Ingest a document (path is relative to the container's /app dir)
curl -X POST http://localhost:8000/api/v1/ingest -H "Content-Type: application/json" -d "{\"path\": \"data/incident_reports/incident_report_001.md\", \"doc_type\": \"incident_report\"}"

# 4. Ask a question
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d "{\"query\": \"What happened in the fall protection incident where a worker removed his harness?\"}"
```

The `curl` commands are single-line and run as-is in **cmd**, bash, and zsh. In **PowerShell**,
`curl` is an alias for `Invoke-WebRequest` — use `curl.exe` instead, or `Invoke-RestMethod`.

The vector store starts empty — ingest at least one document before querying. Repeat step 3 per
file (OSHA PDFs go under `data/osha_docs/`; ten incident reports ship in
`data/incident_reports/`). OSHA PDFs are not committed — download them first (see
[Corpus](#corpus)); they get baked into the image at build time.

Interactive API docs: <http://localhost:8000/docs>.

## API

All routes are mounted under `/api/v1`.

| Method | Path | Body / description |
|---|---|---|
| `POST` | `/query` | `{ "query": str, "session_id"?: str }` — returns `answer` + cited `sources`. `session_id` keeps conversational memory. |
| `POST` | `/ingest` | `{ "path": str, "doc_type"?: str }` — ingest one `.pdf` or `.md`/`.markdown` file. `doc_type` defaults to `"regulation"`. |
| `GET` | `/sources` | List indexed documents with their `doc_type` and chunk count. |

`doc_type` filtering is agent-driven: the model decides whether to scope retrieval to
regulations or incident reports through the `retrieve_safety_docs` tool.

## Configuration

Settings load from `.env` (see `app/config.py`). Required keys have no default — the app will
not start without them.

| Variable | Default | Notes |
|---|---|---|
| `NVIDIA_API_KEY` | — (required) | Embeddings |
| `COHERE_API_KEY` | — (required) | Reranking |
| `GOOGLE_API_KEY` | — (required) | Chat model (Gemini) |
| `QDRANT_URL` | `http://localhost:6333` | Overridden to `http://qdrant:6333` under compose |
| `QDRANT_COLLECTION_NAME` | `safety-docs` | |
| `EMBEDDING_MODEL` | `nvidia/llama-nemotron-embed-1b-v2` | |
| `COHERE_RERANK_MODEL` | `rerank-v4.0-pro` | |
| `CHAT_MODEL` | `gemini-2.5-flash` | Chat model id |
| `CHAT_MODEL_PROVIDER` | `google_genai` | `init_chat_model` provider |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | |
| `RETRIEVAL_K` / `TOP_N` | `6` / `4` | Candidates retrieved / kept after rerank |

## Evaluation

Offline eval over a 15-question hand-written set (`eval/eval_dataset.json`), measuring whether
the expected source document is retrieved and how faithful the generated answer is.

| Metric | Score | Dataset |
|---|---|---|
| Hit-rate@1 | 93.3% | 15 questions |
| Hit-rate@3 | 100% | 15 questions |
| Hit-rate@5 | 100% | 15 questions |
| MRR | 0.967 | 15 questions |

Retrieval numbers are from the run on 2026-07-11 (`eval/results/`). Answer faithfulness is
opt-in (`--faithfulness`) and runs the full agent plus an LLM judge per question.

```bash
python -m eval.run_eval                # retrieval metrics only
python -m eval.run_eval --faithfulness # also run the LLM-as-judge faithfulness pass
```

Requires Qdrant running with the corpus already ingested, plus the API keys in `.env`.

## Corpus

- **Incident reports** — ten synthetic Markdown field reports ship in `data/incident_reports/`.
- **OSHA regulations** — PDFs are not committed. Download free from
  [osha.gov/publications](https://www.osha.gov/publications) and
  [29 CFR 1926](https://www.osha.gov/laws-regs/regulations/standardnumber/1926) into
  `data/osha_docs/`, then ingest them via `POST /ingest` with `"doc_type": "regulation"`.

## Local development

Without Docker (needs a reachable Qdrant, e.g. `docker compose up -d qdrant`):

```bash
uv sync                                        # install dependencies
uvicorn app.main:app --reload --port 8000      # run the API
```

## Project layout

```
app/
  api/routes.py        FastAPI endpoints (/query, /ingest, /sources)
  core/ingestion.py    PDF/Markdown extraction, chunking, async Qdrant indexing
  core/rag_chain.py    Retrieval + rerank + agent with memory
  core/reranker.py     Cohere rerank with rate-limit backoff
  models/schemas.py    Pydantic request/response models
  config.py            Settings via pydantic-settings
  main.py              App entrypoint
data/                  Corpus (incident reports committed, OSHA PDFs downloaded)
eval/                  Evaluation dataset, runner, and results
```

## License

No license file yet — usage terms are unspecified until one is added.

# MY RAG

This repository contains a medical retrieval-augmented generation (RAG) pipeline for indexing research PDFs and answering questions against the built hybrid FAISS/keyword index.

## Project Layout

```text
.
├── rag_system/              # Application package
│   ├── ingestion/           # PDF loading, chunking, embedding helpers
│   ├── retrieval/           # Retrieval/reranking modules
│   └── utils/               # Configuration and preprocessing utilities
├── datasets/
│   ├── raw/pdfs/            # Source PDF documents
│   └── questions/           # Batch question workbooks
├── outputs/
│   ├── index/               # Generated FAISS and metadata index artifacts
│   └── reports/             # Generated Excel reports
├── docs/                    # Notes and project documentation
├── scripts/                 # Operational scripts
├── tests/                   # Test suite
└── requirements.txt         # Python dependencies
```

The source package remains at the repository root so existing imports such as `from rag_system.qa_cli import answer_question` continue to work.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a local `.env` file for secrets such as the Groq API key. The `.env` file is intentionally ignored by Git.

## Build the Index

```powershell
python DataIngestion.py
```

By default this reads PDFs from `datasets/raw/pdfs` and writes index artifacts to `outputs/index/faiss_index`. You can override the paths:

```powershell
python DataIngestion.py --pdf-dir datasets/raw/pdfs --output-dir outputs/index
```

## Query the Index

Ask one question:

```powershell
python -m rag_system.qa_cli "Which papers mention COPD?"
```

Run interactive mode:

```powershell
python -m rag_system.qa_cli
```

## Batch Questions

The batch runner reads `datasets/questions/questions.xlsx` and writes `outputs/reports/rag_results.xlsx`.

```powershell
python -m rag_system.qa_bulkload
```

## Run the HTTP API

After building the index, start the production API:

```powershell
python scripts/run_api.py
```

Available endpoints:

- `GET /health`
- `GET /ready`
- `GET /papers`
- `POST /query`

Example query:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"question":"Which papers mention COPD?","top_k":5}'
```

Docker Compose is also available:

```powershell
docker compose up --build
```

## Configuration

Runtime defaults live in `rag_system/utils/config.py`. Environment variables with the `RAG_` prefix can override key settings, including:

- `RAG_PDF_DIR`
- `RAG_OUTPUT_DIR`
- `RAG_MODEL_NAME`
- `RAG_TOP_K`
- `RAG_CHUNK_SIZE`
- `RAG_RETRIEVAL_ALPHA`
- `RAG_ENABLE_CACHE`
- `RAG_API_HOST`
- `RAG_API_PORT`
- `RAG_LOG_LEVEL`
- `RAG_JSON_LOGS`

Legacy paths such as `pdfs`, `output`, and root-level `questions.xlsx` are still detected as fallbacks when the organized paths do not exist.

## Production Hardening

The repository now includes:

- application service layer: `rag_system.service.RAGService`
- structured logging: `rag_system.logging_config`
- bounded TTL caches: `rag_system.cache`
- telemetry counters and latency summaries: `rag_system.telemetry`
- FastAPI service: `rag_system.api`
- evaluation helpers: `rag_system.evaluation`
- benchmark helpers: `rag_system.benchmark`
- Docker, Docker Compose, and GitHub Actions CI

See [ARCHITECTURE.md](ARCHITECTURE.md), [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md), [CONFIGURATION.md](CONFIGURATION.md), [DEPLOYMENT.md](DEPLOYMENT.md), [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md), and [PERFORMANCE.md](PERFORMANCE.md).

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

## Configuration

Runtime defaults live in `rag_system/utils/config.py`. Environment variables with the `RAG_` prefix can override key settings, including:

- `RAG_PDF_DIR`
- `RAG_OUTPUT_DIR`
- `RAG_MODEL_NAME`
- `RAG_TOP_K`
- `RAG_CHUNK_SIZE`

Legacy paths such as `pdfs`, `output`, and root-level `questions.xlsx` are still detected as fallbacks when the organized paths do not exist.

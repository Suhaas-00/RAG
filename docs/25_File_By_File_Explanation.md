# File By File Explanation

## Table of Contents
- [Root](#root)
- [rag_system](#rag_system)
- [ingestion](#ingestion)
- [retrieval](#retrieval)
- [utils](#utils)
- [Artifacts](#artifacts)

## Root

| File | Purpose |
| --- | --- |
| `DataIngestion.py` | Main ingestion pipeline and CLI. |
| `README.md` | Short setup and usage guide. |
| `requirements.txt` | Dependency list. |
| `.env` | Local Groq key variable; secret omitted. |
| `.gitignore` | Git ignore rules. |

## rag_system

| File | Purpose |
| --- | --- |
| `qa_cli.py` | Active QA CLI and orchestration. |
| `qa_bulkload.py` | Excel batch inference. |
| `llm.py` | Groq prompt and API wrapper. |
| `hybrid_retriever.py` | Active hybrid retrieval. |
| `reranker.py` | Active optional CrossEncoder reranker. |
| `query_parser.py` | Query intent parsing. |
| `metadata.py` | Metadata extraction and record normalization. |
| `cleaner.py` | PDF cleanup and sections. |

## ingestion

| File | Purpose |
| --- | --- |
| `chunking.py` | Content and metadata chunk generation. |
| `embedding.py` | Embedding model wrapper. |
| `DataIngestion.py` | Compatibility wrapper to root ingestion module. |
| `ingest_cli.py` | Module CLI shim. |

## retrieval

| File | Purpose |
| --- | --- |
| `retriever.py` | Alternate retriever using stored TF-IDF. |
| `hybrid_search.py` | Alternate standalone hybrid search helpers. |
| `reranker.py` | Alternate section-aware reranking. |

## utils

| File | Purpose |
| --- | --- |
| `config.py` | `Settings` dataclass and validation. |
| `preprocessing.py` | Shared cleaning, token counting, metadata regex extraction. |
| `__init__.py` | `debug_dump`. |

## Artifacts

`datasets/raw/pdfs` stores source PDFs. `outputs/index/faiss_index` stores active vector and payload files. `outputs/index/*.json` stores readable manifests. `datasets/questions/questions.xlsx` stores batch questions.

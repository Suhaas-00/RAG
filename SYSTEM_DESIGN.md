# System Design

## Components

- `DataIngestion.py`: offline index builder for PDFs.
- `rag_system.ingestion`: chunking, cleaning, embedding helpers.
- `rag_system.hybrid_retriever`: primary runtime retriever used by CLI, batch, service, and API.
- `rag_system.retrieval`: lower-level and compatibility retrieval modules.
- `rag_system.service`: application boundary for RAG requests.
- `rag_system.api`: HTTP API factory.
- `rag_system.cache`: in-process TTL caches.
- `rag_system.telemetry`: in-memory counters and latency summaries.
- `rag_system.evaluation`: retrieval and answer regression checks.
- `rag_system.benchmark`: latency benchmark helper.

## Request Lifecycle

1. A question enters through CLI, batch job, or HTTP API.
2. `RAGService` parses intent and builds a retrieval query.
3. List-papers and metadata-only queries are answered without dense retrieval.
4. Other questions pass through retrieval cache lookup.
5. `HybridRetriever` applies metadata and document-scope filtering.
6. Dense FAISS scores and lexical BM25-style scores are combined.
7. `CrossEncoderReranker` reranks candidates when available and falls back deterministically when not.
8. Context is formatted with citations.
9. Groq generation produces the final grounded response, or the retrieved context is returned when no API key is configured.
10. Metrics, latency, and cache statistics are updated.

## Compatibility

Existing commands remain supported:

```powershell
python DataIngestion.py
python -m rag_system.qa_cli "Which papers mention COPD?"
python -m rag_system.qa_bulkload
```

New production entry point:

```powershell
python scripts/run_api.py
```


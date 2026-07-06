# Performance Guide

## Current Optimizations

- FAISS `IndexFlatIP` over normalized embeddings.
- BM25-style lexical scoring for exact term recovery.
- Cross-encoder reranking with deterministic fallback.
- In-process bounded TTL caches for retrieval and responses.
- Query short-circuiting for list-papers and metadata-only requests.
- Metrics for operation counts and latency summaries.

## Benchmarking

Use `rag_system.benchmark.benchmark_questions` from a Python script or notebook:

```python
from rag_system.benchmark import benchmark_questions
from rag_system.service import RAGService

service = RAGService.from_settings()
result = benchmark_questions(service, ["list papers"], use_cache=False)
print(result)
```

## Scaling Guidance

- Use a persistent external cache when running multiple API replicas.
- Move ingestion to a separate worker process for large corpora.
- Use GPU-backed embedding/reranking hosts when latency matters.
- Export telemetry to OpenTelemetry or the platform metrics backend in production clusters.


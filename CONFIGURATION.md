# Configuration Guide

All runtime settings are centralized in `rag_system.utils.config.Settings`.

## Important Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `RAG_MODEL_NAME` | Embedding model name | `NeuML/pubmedbert-base-embeddings` |
| `RAG_PDF_DIR` | Source PDF directory | `datasets/raw/pdfs` |
| `RAG_OUTPUT_DIR` | Index output directory | `outputs/index` |
| `RAG_TOP_K` | Default retrieved chunk count | `3` |
| `RAG_RETRIEVAL_ALPHA` | Dense/lexical blend for service APIs | `0.55` |
| `RAG_ENABLE_DOCUMENT_SCOPE` | Enable document-scoped filtering | `false` |
| `RAG_ALLOW_GLOBAL_SEARCH` | Permit global retrieval when unscoped | `true` |
| `RAG_ENABLE_CACHE` | Enable service caches | `true` |
| `RAG_CACHE_MAX_SIZE` | Entries per cache | `512` |
| `RAG_RETRIEVAL_CACHE_TTL_SECONDS` | Retrieval cache TTL | `300` |
| `RAG_RESPONSE_CACHE_TTL_SECONDS` | Response cache TTL | `300` |
| `RAG_API_HOST` | API bind host | `0.0.0.0` |
| `RAG_API_PORT` | API bind port | `8000` |
| `RAG_LOG_LEVEL` | Log level | `INFO` |
| `RAG_JSON_LOGS` | Emit JSON logs | `false` |

## Validation

`Settings` validates ranges for chunking, retrieval weights, cache sizes, cache TTLs, API port, and log level at construction time.


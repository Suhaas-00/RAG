# Configuration

## Table of Contents
- [Settings](#settings)
- [Defaults](#defaults)
- [Validation](#validation)
- [Environment Constructor](#environment-constructor)
- [CLI Defaults](#cli-defaults)

## Settings

`Settings` is a frozen dataclass in `rag_system.utils.config`.

## Defaults

| Field | Default |
| --- | --- |
| `model_name` | `NeuML/pubmedbert-base-embeddings` |
| `pdf_dir` | `datasets/raw/pdfs` unless legacy fallback applies |
| `output_dir` | `outputs/index` unless legacy fallback applies |
| `chunk_size` | `400` |
| `chunk_overlap` | `64` |
| `embedding_batch_size` | `32` |
| `semantic_weight` | `0.7` |
| `keyword_weight` | `0.3` |
| `candidate_k` | `10` |
| `top_k` | `3` |
| `max_context_tokens` | `2000` |
| `min_chunk_tokens` | `40` |

Derived paths are `index_dir`, `metadata_path`, and `vectors_path`.

## Validation

Settings validates positive sizes, overlap bounds, weights in `[0,1]`, `candidate_k >= top_k`, and positive context/chunk/batch values.

## Environment Constructor

`Settings.from_env` reads `RAG_MODEL_NAME`, `RAG_PDF_DIR`, `RAG_OUTPUT_DIR`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_EMBEDDING_BATCH`, `RAG_SEMANTIC_WEIGHT`, `RAG_KEYWORD_WEIGHT`, `RAG_CANDIDATE_K`, `RAG_TOP_K`, `RAG_MAX_CONTEXT_TOKENS`, and `RAG_MIN_CHUNK_TOKENS`. Active CLIs do not call this method.

## CLI Defaults

Ingestion parser defaults mostly match `Settings`. QA parser defaults differ: `top_k=5` and `alpha=0.55`.

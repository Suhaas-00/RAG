# Current Limitations

## Table of Contents
- [Retrieval](#retrieval)
- [Chunking](#chunking)
- [Prompting](#prompting)
- [Embeddings](#embeddings)
- [Configuration](#configuration)
- [Testing](#testing)

## Retrieval

The active retriever does not use stored TF-IDF artifacts, supports fewer metadata filters than the alternate retriever, has no score threshold, and rebuilds a temporary FAISS subset index per query.

## Chunking

Chunking uses regex sentence splitting and regex token estimates. It does not perform OCR, semantic chunk boundary detection, structured table parsing, or model-tokenizer budgeting.

## Prompting

The prompt requires citations and grounding, but the code does not validate citations or run an answer verifier.

## Embeddings

Only model name and vector dimension are recorded. Device selection is delegated to `SentenceTransformer`. Any preprocessing/model change requires reindexing.

## Configuration

`Settings.from_env` exists but is not used by active CLIs. `max_context_tokens` exists but is not enforced. QA CLI defaults differ from `Settings` defaults.

## Testing

The `tests` directory is empty. No automated coverage exists in this checkout.

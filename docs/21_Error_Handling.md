# Error Handling

## Table of Contents
- [Ingestion](#ingestion)
- [Configuration](#configuration)
- [Embedding](#embedding)
- [Retrieval](#retrieval)
- [LLM](#llm)
- [Batch](#batch)

## Ingestion

Missing PDF directories, no PDFs, and no usable chunks are fatal. Unreadable individual PDFs are skipped with warnings.

## Configuration

`Settings` and `chunk_pages` raise `ValueError` for invalid numeric ranges or invalid source values.

## Embedding

`PubMedEmbedder` raises `ImportError` if `sentence-transformers` is absent. `encode` raises `ValueError` for empty text lists, invalid batch size, or chunk type length mismatch.

## Retrieval

`HybridRetriever.load` raises `FileNotFoundError` when required files are missing. `qa_cli.main` catches any load exception and exits with an error message.

## LLM

Missing API key is handled by fallback. Groq API exceptions are not caught inside `answer_with_groq`.

## Batch

`run_bulk_inference` catches exceptions per question, logs them, and writes an explicit error response row.

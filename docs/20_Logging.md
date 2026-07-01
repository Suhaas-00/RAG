# Logging

## Table of Contents
- [Setup](#setup)
- [Ingestion](#ingestion)
- [QA CLI](#qa-cli)
- [Bulk Runner](#bulk-runner)
- [Verbose Debug](#verbose-debug)

## Setup

Modules define loggers with `logging.getLogger(__name__)`. Entry points configure handlers.

## Ingestion

`DataIngestion.main` uses timestamped logging with level from `--log-level`. It logs discovery, metadata summaries, chunk counts, embedding, FAISS, TF-IDF, output paths, and elapsed time.

## QA CLI

`qa_cli.main` configures logging with format `%(levelname)s %(name)s: %(message)s` and default level `WARNING`.

## Bulk Runner

`qa_bulkload.py` calls `logging.basicConfig` at import time with level `INFO`. It logs workbook loading, index loading, progress, failures, and summary statistics.

## Verbose Debug

`debug_dump` prints JSON payloads only when enabled. `qa_cli` uses it for parsed query and retrieval debug output under `--verbose`.

# Code Walkthrough

## Table of Contents
- [Ingestion Flow](#ingestion-flow)
- [QA Flow](#qa-flow)
- [Bulk Flow](#bulk-flow)
- [Shared Utilities](#shared-utilities)
- [Alternate Retrieval](#alternate-retrieval)

## Ingestion Flow

`DataIngestion.main` parses args, configures logging, creates `Settings`, and calls `ingest`. `ingest` discovers PDFs, builds paper labels, loads pages, extracts metadata, chunks pages, normalizes and filters records, wires neighbors, embeds chunks, creates FAISS and TF-IDF structures, and writes artifacts.

## QA Flow

`qa_cli.main` loads `HybridRetriever` and calls `answer_question`. `answer_question` parses intent, handles list/metadata shortcuts, retrieves chunks, optionally returns direct section text, or calls Groq.

## Bulk Flow

`qa_bulkload.run_bulk_inference` loads questions, index, and paper map once, processes each question with parser/retriever/LLM functions, and writes detailed and summary sheets.

## Shared Utilities

`normalize_for_embedding` is the shared ingestion/query contract. `cleaner.py` owns page cleanup and section detection. `metadata.py` owns paper mapping and record schema normalization.

## Alternate Retrieval

`rag_system.retrieval` contains older or alternate retrieval utilities. They are present and documented, but `qa_cli` imports the root-level active retriever.

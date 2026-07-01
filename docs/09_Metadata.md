# Metadata

## Table of Contents
- [Metadata Creation](#metadata-creation)
- [Document Fields](#document-fields)
- [Chunk Fields](#chunk-fields)
- [Storage](#storage)
- [Usage](#usage)
- [Current Payload](#current-payload)

## Metadata Creation

The main ingestion path uses `rag_system.metadata.extract_document_metadata`, not the fallback helper in `rag_system.utils.preprocessing`, because `DataIngestion.ingest` passes `doc_metadata` into `chunk_pages`.

## Document Fields

| Field | Source |
| --- | --- |
| `source` | PDF filename |
| `paper_id` | sorted `paper N` label |
| `doi` | regex over first 5000 chars |
| `pmid` | PMID/PubMed ID regex over full text |
| `year` | first plausible year in first 5000 chars |
| `diseases` | disease regex over full text |
| `genes` | gene/biomarker regex over full text |
| `study_designs` | study design regex over full text |

## Chunk Fields

`normalize_record` standardizes `source`, `paper_id`, `page`, `page_number`, `section`, `chunk_id`, `previous_chunk`, `next_chunk`, `prev_chunk_id`, `next_chunk_id`, `token_count`, `chunk_type`, and nested `metadata`.

## Storage

Metadata is stored in `metadata.pkl`, `chunks_manifest.json`, and `doc_metadata_index.json`. `doc_metadata_index` is keyed by filename, filename stem, and paper label.

## Usage

`qa_cli._metadata_answer` reads document metadata directly. `query_parser` creates filters. `HybridRetriever._metadata_filter` filters records. `format_context` emits metadata in context headers. `qa_bulkload` writes metadata columns into Excel.

## Current Payload

Current payload facts: schema `3`, model `NeuML/pubmedbert-base-embeddings`, dimension `768`, records `290`, content chunks `225`, metadata chunks `65`, TF-IDF shape `(290, 30111)`, and 11 paper labels.

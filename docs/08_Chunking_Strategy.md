# Chunking Strategy

## Table of Contents
- [Overview](#overview)
- [Content Chunks](#content-chunks)
- [Metadata Chunks](#metadata-chunks)
- [IDs and Provenance](#ids-and-provenance)
- [Overlap](#overlap)
- [Validation](#validation)
- [Schema](#schema)
- [Tradeoffs](#tradeoffs)

## Overview

`chunk_pages` creates content chunks and metadata chunks, then returns them as one list. Defaults are `chunk_size=400` and `overlap=64`.

## Content Chunks

The content path is `clean_pdf_pages -> _pages_to_units -> _split_into_sentences -> _units_to_chunks`. Units are `(sentence, section, page_number)` triples. `_units_to_chunks` packs sentences until the token budget is reached or the section changes. A single sentence is always included even if it exceeds the budget.

## Metadata Chunks

`_build_metadata_chunks` creates short chunks for populated DOI, PubMed ID, publication year, journal, diseases, genes, study designs, and abstract. Field chunks are formatted as `<FIELD>: <value>` and capped at 300 estimated tokens. Abstract uses detected abstract lines from the first three pages, or a page-1 pseudo-abstract fallback.

## IDs and Provenance

Chunk IDs are deterministic 20-character SHA-256 prefixes from `source`, `ordinal`, and the first 80 characters of text. Content chunks use the starting page. Metadata chunks use page 1.

## Overlap

Overlap is applied only within the same section. Section boundaries reset the window. `wire_neighbors` links surviving content chunks after filtering and does not link metadata chunks.

## Validation

`chunk_pages` requires `chunk_size` between 50 and 2000, `overlap >= 0`, `overlap < chunk_size`, `overlap <= 50% of chunk_size`, and non-empty source.

## Schema

Current manifest records contain `chunk_id`, `text`, `section`, `chunk_type`, `source`, `page_number`, `page`, `paper_id`, `prev_chunk_id`, `next_chunk_id`, `previous_chunk`, `next_chunk`, `ordinal`, `token_count`, and nested `metadata`.

## Tradeoffs

This strategy preserves sections and sentence boundaries and indexes structured metadata. It does not use semantic boundary detection, model-tokenizer token counts, OCR, or table extraction.

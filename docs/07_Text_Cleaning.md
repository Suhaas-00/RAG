# Text Cleaning

## Table of Contents
- [Modules](#modules)
- [Shared Cleaning](#shared-cleaning)
- [PDF Line Cleaning](#pdf-line-cleaning)
- [Chunk Cleaning](#chunk-cleaning)
- [Query Cleaning](#query-cleaning)
- [Non-Implemented Cleaning](#non-implemented-cleaning)

## Modules

`rag_system.utils.preprocessing` provides shared cleaning and embedding normalization. `rag_system.cleaner` provides PDF-specific page and line cleanup.

## Shared Cleaning

`clean_text` applies NFKC normalization, soft hyphen removal, CR normalization, PDF hyphen-line rejoining, control-character replacement, inline numeric citation removal, repeated punctuation collapse, whitespace collapse, and optional lowercasing.

`normalize_for_embedding` is the canonical embedding/query function and calls `clean_text(..., lowercase=True)`.

## PDF Line Cleaning

`remove_repeated_headers_footers` counts the first three and last three normalized lines per page and removes repeated candidates. `_line_quality` rejects short lines, stop sections, figure/table lines, boilerplate, short affiliation/author lines on early pages, numeric-heavy lines, and lines with alphabetic ratio below `0.35`.

## Chunk Cleaning

`clean_chunk_text` normalizes text, removes numeric citations, removes figure/table sentences, collapses whitespace, and strips.

## Query Cleaning

`build_retrieval_query` removes paper references, PDF filenames, and common command words. Embedding and lexical retrieval then apply `normalize_for_embedding`.

## Non-Implemented Cleaning

There is no semantic deduplication, no full reference parser, no affiliation metadata extraction, and no author/title extraction in active metadata.

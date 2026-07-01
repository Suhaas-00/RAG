# Query Processing

## Table of Contents
- [QueryIntent](#queryintent)
- [Paper Resolution](#paper-resolution)
- [Section Detection](#section-detection)
- [Intent Order](#intent-order)
- [Retrieval Query](#retrieval-query)

## QueryIntent

`QueryIntent` contains `intent`, `query`, `paper_label`, `paper_source`, `section`, `disease`, `metadata_field`, and `filters`.

## Paper Resolution

The parser recognizes `paper <number>`, explicit `*.pdf` filenames, and 6-10 digit tokens that match PDF filename stems.

## Section Detection

Section detection scans aliases from `CANONICAL_SECTIONS` and returns canonical labels through `canonical_section`.

## Intent Order

The parser checks list-papers, metadata fields, paper+section lookup, paper lookup, disease lookup, section lookup, then semantic QA.

## Retrieval Query

`build_retrieval_query` lowercases the query, strips paper references, strips PDF filenames, removes common command words, collapses whitespace, and prepends section if needed.

# Performance

## Table of Contents
- [Indexing](#indexing)
- [Retrieval](#retrieval)
- [Memory](#memory)
- [Current Artifacts](#current-artifacts)
- [Optimization](#optimization)

## Indexing

Indexing is dominated by PDF extraction, transformer embedding, FAISS vector storage, and TF-IDF fitting. Chunking is approximately linear in cleaned text length.

## Retrieval

Active retrieval scans all records for filters, reconstructs candidate vectors, builds a temporary FAISS index, computes lexical scores over candidates, optionally runs CrossEncoder predictions, and calls Groq.

## Memory

`IndexFlatIP` stores full float32 vectors. With 290 records and 768 dimensions, raw vector storage is about 0.85 MB before FAISS overhead.

## Current Artifacts

| Artifact | Size in checkout |
| --- | --- |
| `vectors.index` | 890,925 bytes |
| `metadata.pkl` | 2,216,097 bytes |
| `chunks_manifest.json` | 707,625 bytes |
| `doc_metadata_index.json` | 19,529 bytes |

## Optimization

Use stored TF-IDF in the active retriever, cache common metadata subsets, add thresholds, enforce context budgets, and consider approximate FAISS indexes for larger corpora.

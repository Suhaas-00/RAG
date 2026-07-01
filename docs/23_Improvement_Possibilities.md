# Improvement Possibilities

## Table of Contents
- [Retrieval](#retrieval)
- [Chunking](#chunking)
- [Metadata](#metadata)
- [LLM](#llm)
- [Evaluation](#evaluation)
- [Operations](#operations)

## Retrieval

Add active disease/gene/year filters, use persisted TF-IDF or BM25, add score thresholds, hybrid candidate merging, and query rewriting.

## Chunking

Use tokenizer-based counts, semantic boundaries, better section parsing, structured table extraction, and OCR for scanned PDFs.

## Metadata

Extract title and authors because the parser already recognizes those fields. Add drug, trial, endpoint, and normalized biomedical entity metadata.

## LLM

Add retries, timeouts, max token controls, citation validation, and answer-grounding checks.

## Evaluation

Create golden QA data, measure recall@k/MRR, validate metadata queries, and add unit tests for parser/chunker/retriever behavior.

## Operations

Remove secrets from local files before sharing, add reproducible index scripts, document model caches, and add CI because `tests/` is empty.

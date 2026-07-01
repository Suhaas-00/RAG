# Retrieval Pipeline

## Table of Contents
- [Active Retriever](#active-retriever)
- [Loading](#loading)
- [Filtering](#filtering)
- [Dense Scores](#dense-scores)
- [Lexical Scores](#lexical-scores)
- [Hybrid Scores](#hybrid-scores)
- [Alternate Retriever](#alternate-retriever)

## Active Retriever

`qa_cli.py` and `qa_bulkload.py` use `rag_system.hybrid_retriever.HybridRetriever`.

## Loading

`HybridRetriever.load` reads FAISS and payload files, initializes `PubMedEmbedder` with the payload model name, and constructs `CrossEncoderReranker`.

## Filtering

Active filters are `source`, `paper_id`, `section`, and `chunk_type`. Filtering is a full scan over payload records using exact or case-insensitive exact comparisons.

## Dense Scores

The retriever encodes the query, reconstructs candidate vectors, builds a temporary `IndexFlatIP`, and searches up to `candidate_k` candidates.

## Lexical Scores

`_lexical_scores` computes BM25-style scores over the filtered candidate set with `k1=1.5` and `b=0.75`, then normalizes by the maximum raw score. The active retriever does not use the stored TF-IDF matrix.

## Hybrid Scores

The active formula is `alpha * dense + (1 - alpha) * lexical`. `qa_cli` defaults `alpha=0.55`.

## Alternate Retriever

`rag_system.retrieval.retriever.RAGRetriever` is present but not used by `qa_cli`. It uses stored TF-IDF rows and supports disease, gene, and year filters in addition to source, section, and chunk type.

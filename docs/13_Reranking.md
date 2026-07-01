# Reranking

## Table of Contents
- [Active Reranker](#active-reranker)
- [CrossEncoder Mode](#crossencoder-mode)
- [Fallback Mode](#fallback-mode)
- [Confidence](#confidence)
- [Alternate Reranker](#alternate-reranker)

## Active Reranker

The active class is `rag_system.reranker.CrossEncoderReranker`.

## CrossEncoder Mode

The constructor attempts to load `cross-encoder/ms-marco-MiniLM-L-6-v2` through `sentence_transformers.CrossEncoder`. If loaded, it scores `(query, candidate_text)` pairs with `model.predict`.

## Fallback Mode

If the CrossEncoder import or load fails, the exception is caught. The model remains `None`, and reranking uses existing `hybrid_score` or `final_score`.

## Confidence

Confidence is a weighted blend of rerank component and hybrid component: `0.65 * rerank_component + 0.35 * hybrid_component`, rounded to three decimals.

## Alternate Reranker

`rag_system.retrieval.reranker` implements deterministic section-aware reranking with section boosts. It is not used by active `qa_cli`.

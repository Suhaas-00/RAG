# Class Reference

## Table of Contents
- [Settings](#settings)
- [PageText and CleanPage](#pagetext-and-cleanpage)
- [PubMedEmbedder](#pubmedembedder)
- [QueryIntent](#queryintent)
- [HybridRetriever and RetrievalResult](#hybridretriever-and-retrievalresult)
- [CrossEncoderReranker](#crossencoderreranker)
- [Alternate RAGRetriever](#alternate-ragretriever)

## Settings

`Settings` is an immutable dataclass. It owns ingestion and retrieval defaults plus derived artifact paths. Its constructor validates bounds immediately through `__post_init__`. Use `to_dict` when serializing into payloads. `from_env` exists for `RAG_*` variables, but active CLIs do not call it.

## PageText and CleanPage

`PageText` is the ingestion page container with fields `text` and `page_number`. It validates that `page_number >= 1`. `CleanPage` is the cleaned page container with fields `text` and `page`.

## PubMedEmbedder

`PubMedEmbedder` loads a SentenceTransformer model and exposes `dimension`, `encode`, `encode_chunks`, and `encode_query`. It is responsible for shared normalization, optional chunk-type prefixes, model encoding, and defensive L2 normalization.

## QueryIntent

`QueryIntent` is a frozen dataclass produced by `parse_query`. It carries the raw cleaned query, resolved paper references, section, disease, metadata field, and filters passed to retrieval.

## HybridRetriever and RetrievalResult

`HybridRetriever` is the active retriever. Constructor inputs are loaded FAISS index, payload, embedder, and optional alpha. It owns payload records, paper map, and active reranker. `retrieve` returns active `RetrievalResult`, whose fields are `context`, `chunks`, and `debug`.

## CrossEncoderReranker

`CrossEncoderReranker` attempts to load a CrossEncoder model. It degrades gracefully by using hybrid scores when the model is unavailable.

## Alternate RAGRetriever

`rag_system.retrieval.retriever.RAGRetriever` is an alternate class with a similar role but different implementation. It uses stored TF-IDF matrix for keyword scoring and has a `RetrievalResult` class without a `debug` field. It is not the active `qa_cli` retriever.

# Embedding Model

## Table of Contents
- [Implementation](#implementation)
- [Model](#model)
- [Normalization](#normalization)
- [Prefixes](#prefixes)
- [Batching and Device](#batching-and-device)
- [Known and Unknown Specifications](#known-and-unknown-specifications)

## Implementation

`PubMedEmbedder` wraps `sentence_transformers.SentenceTransformer` and is shared by ingestion and retrieval.

## Model

The code default and current payload model is `NeuML/pubmedbert-base-embeddings`. The current payload records embedding dimension `768`.

## Normalization

Texts are normalized by `normalize_for_embedding`. `SentenceTransformer.encode` is called with `normalize_embeddings=True`, then vectors are L2-normalized again with NumPy. FAISS inner product therefore acts as cosine similarity.

## Prefixes

With `use_prefixes=True`, metadata chunks receive `medical document metadata: ` and content chunks receive `medical document passage: `. Queries use `encode_query` and receive no chunk prefix.

## Batching and Device

Default batch size is 32. Device is optional and passed to `SentenceTransformer`; explicit GPU/CPU selection is not implemented by the project configuration.

## Known and Unknown Specifications

The repository records model name and dimension. It does not record maximum sequence length, pooling strategy, training objective, memory usage, or speed. Those must be verified from the installed model/runtime, not assumed from this codebase.

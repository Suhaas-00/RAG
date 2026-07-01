# Vector Database

## Table of Contents
- [FAISS](#faiss)
- [Index Type](#index-type)
- [Metric](#metric)
- [Files](#files)
- [Loading](#loading)
- [Limitations](#limitations)

## FAISS

The project uses `faiss-cpu`. No external vector database service is used.

## Index Type

Ingestion creates `faiss.IndexFlatIP(embedder.dimension)` and adds all normalized vectors. This is exact flat search, not approximate search.

## Metric

The metric is inner product. Because vectors are unit-normalized, this is cosine similarity in practice.

## Files

| File | Role |
| --- | --- |
| `outputs/index/faiss_index/vectors.index` | Active FAISS index. |
| `outputs/index/faiss_index/metadata.pkl` | Active payload. |
| `outputs/index/faiss_index/index.faiss` | Legacy vector artifact; active loader uses it only if `vectors.index` is absent. |
| `outputs/index/faiss_index/index.pkl` | Legacy artifact not read by active code. |

## Loading

`HybridRetriever.load` reads `vectors.index` or fallback `index.faiss`, then unpickles `metadata.pkl`.

## Limitations

`IndexFlatIP` stores all vectors and performs exact search. Active filtered search reconstructs candidate vectors and builds a temporary sub-index per query, which is simple but not optimal for large corpora.

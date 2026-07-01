# Model Specifications

## Table of Contents
- [Embedding Model](#embedding-model)
- [Reranker Model](#reranker-model)
- [LLM](#llm)

## Embedding Model

The default and current payload embedding model is `NeuML/pubmedbert-base-embeddings`, loaded by `sentence_transformers.SentenceTransformer`. The current payload records dimension `768`.

The repository does not define maximum sequence length, pooling strategy, training objective, runtime speed, or memory usage. Those values are not assumed in this documentation.

## Reranker Model

The active reranker attempts to load `cross-encoder/ms-marco-MiniLM-L-6-v2`. If unavailable, it falls back to hybrid scores.

## LLM

The default Groq model is `llama-3.1-8b-instant`. The only explicit generation parameter is `temperature=0.1`. The code does not set top-p, max tokens, context window, streaming, retries, or timeouts.

# Appendix

## Table of Contents
- [Glossary](#glossary)
- [Abbreviations](#abbreviations)
- [LLM Concepts](#llm-concepts)
- [Embedding Concepts](#embedding-concepts)
- [FAISS Terms](#faiss-terms)
- [Medical NLP Terms](#medical-nlp-terms)

## Glossary

| Term | Meaning |
| --- | --- |
| RAG | Retrieval-augmented generation. |
| Chunk | One retrieval record. |
| Content chunk | Prose chunk from cleaned PDF text. |
| Metadata chunk | Structured field chunk. |
| Payload | `metadata.pkl`, the retrieval metadata bundle. |
| Hybrid score | Dense and lexical score blend. |
| Rerank score | Score after CrossEncoder or fallback reranking. |
| Fallback | `Not found in the document`. |

## Abbreviations

| Abbreviation | Meaning |
| --- | --- |
| FAISS | Facebook AI Similarity Search. |
| TF-IDF | Term Frequency-Inverse Document Frequency. |
| BM25 | Lexical ranking function. |
| DOI | Digital Object Identifier. |
| PMID | PubMed Identifier. |
| NSCLC | Non-Small Cell Lung Cancer. |
| EGFR | Epidermal Growth Factor Receptor. |
| MET | MET proto-oncogene/receptor tyrosine kinase. |
| PFS | Progression-Free Survival. |
| ORR | Objective Response Rate. |

## LLM Concepts

The LLM is used only after retrieval. The repository does not fine-tune or host a local LLM. It calls Groq chat completions with retrieved context.

## Embedding Concepts

`SentenceTransformer` maps text into dense vectors. The project normalizes vectors so inner product equals cosine similarity.

## FAISS Terms

`IndexFlatIP` is an exact inner-product index. `add` stores vectors. `search` retrieves nearest vectors. `reconstruct` retrieves a stored vector by row index.

## Medical NLP Terms

The code uses regex-based entity heuristics for diseases, genes, and study design terms. It does not use a biomedical named-entity recognition model.

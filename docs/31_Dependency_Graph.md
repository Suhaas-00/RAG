# Dependency Graph

## Table of Contents
- [Runtime Dependencies](#runtime-dependencies)
- [Internal Module Graph](#internal-module-graph)
- [External Package Usage](#external-package-usage)

## Runtime Dependencies

```mermaid
flowchart TD
    DataIngestion --> faiss
    DataIngestion --> numpy
    DataIngestion --> pypdf
    DataIngestion --> sklearn
    DataIngestion --> chunking
    DataIngestion --> embedding
    DataIngestion --> metadata
    DataIngestion --> config
    DataIngestion --> preprocessing
    qa_cli --> hybrid_retriever
    qa_cli --> llm
    qa_cli --> query_parser
    qa_cli --> utils
    qa_bulkload --> pandas
    qa_bulkload --> hybrid_retriever
    qa_bulkload --> llm
    hybrid_retriever --> faiss
    hybrid_retriever --> numpy
    hybrid_retriever --> embedding
    hybrid_retriever --> reranker
    embedding --> sentence_transformers
    reranker --> sentence_transformers
    llm --> dotenv
    llm --> groq
```

## Internal Module Graph

```mermaid
flowchart LR
    cleaner --> metadata
    cleaner --> chunking
    preprocessing --> chunking
    preprocessing --> embedding
    preprocessing --> hybrid_retriever
    query_parser --> qa_cli
    query_parser --> qa_bulkload
    hybrid_retriever --> qa_cli
    hybrid_retriever --> qa_bulkload
    llm --> qa_cli
    llm --> qa_bulkload
```

## External Package Usage

| Package | Used? | Location |
| --- | --- | --- |
| `faiss-cpu` | Yes | ingestion and retrieval |
| `numpy` | Yes | embeddings/retrieval |
| `pypdf` | Yes | PDF extraction |
| `scikit-learn` | Yes | ingestion TF-IDF |
| `sentence-transformers` | Yes | embeddings and reranker |
| `torch`, `transformers` | Indirect | sentence-transformers runtime |
| `groq` | Yes | LLM inference |
| `python-dotenv` | Yes | API key loading |
| `pandas`, `openpyxl` | Yes | batch Excel I/O |
| `pdfplumber` | Listed, not imported | none in current code |
| `langchain`, `langchain-community` | Listed, not imported | none in current code |
| `nltk`, `tqdm` | Listed, not imported | none in current code |

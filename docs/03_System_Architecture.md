# System Architecture

## Table of Contents
- [Overall Architecture](#overall-architecture)
- [Indexing Pipeline](#indexing-pipeline)
- [Retrieval Pipeline](#retrieval-pipeline)
- [LLM Pipeline](#llm-pipeline)
- [Configuration Loading](#configuration-loading)
- [CLI Flow](#cli-flow)

## Overall Architecture

```mermaid
flowchart TB
    subgraph Offline[Index Build]
        P[PDFs] --> L[load_pdf]
        L --> C[clean_pdf_pages]
        C --> CH[chunk_pages]
        CH --> Q[_is_quality_chunk]
        Q --> E[PubMedEmbedder.encode_chunks]
        E --> V[vectors.index]
        Q --> M[metadata.pkl]
        Q --> J[chunks_manifest.json]
    end
    subgraph Online[QA]
        U[Question] --> CLI[qa_cli.main]
        CLI --> R[HybridRetriever.load]
        CLI --> I[parse_query]
        I --> H[HybridRetriever.retrieve]
        H --> RR[CrossEncoderReranker]
        RR --> CTX[format_context]
        CTX --> G[answer_with_groq]
    end
    V --> R
    M --> R
```

## Indexing Pipeline

```mermaid
flowchart TD
    A[Discover PDFs] --> B[build_paper_map]
    B --> C[load_pdf]
    C --> D[extract_document_metadata]
    D --> E[chunk_pages]
    E --> F[normalize_record]
    F --> G[quality filter]
    G --> H[wire_neighbors]
    H --> I[encode_chunks]
    I --> J[FAISS IndexFlatIP]
    H --> K[TfidfVectorizer.fit_transform]
    J --> L[write vectors.index]
    K --> M[write metadata.pkl]
    H --> N[write chunks_manifest.json]
```

## Retrieval Pipeline

```mermaid
flowchart TD
    A[Question] --> B[parse_query]
    B --> C[build_retrieval_query]
    B --> D[filters]
    C --> E[HybridRetriever.retrieve]
    D --> F[_metadata_filter]
    F --> G[_dense_scores]
    F --> H[_lexical_scores]
    G --> I[hybrid score]
    H --> I
    I --> J[sort]
    J --> K[CrossEncoderReranker.rerank]
    K --> L[format_context]
```

## LLM Pipeline

```mermaid
sequenceDiagram
    participant CLI as qa_cli.answer_question
    participant RET as HybridRetriever
    participant LLM as llm.answer_with_groq
    participant ENV as .env/process env
    participant GROQ as Groq API
    CLI->>RET: retrieve(retrieval_query, intent)
    RET-->>CLI: RetrievalResult
    CLI->>LLM: answer_with_groq(question, context)
    LLM->>ENV: load_groq_api_key()
    alt key present
        LLM->>GROQ: chat.completions.create
        GROQ-->>LLM: message.content
    else key missing
        LLM-->>CLI: context or fallback
    end
```

## Configuration Loading

`qa_cli` uses `Settings().index_dir` as the default index directory. It does not call `Settings.from_env`. `llm.load_groq_api_key` independently searches upward for `.env` and loads the first one with `override=False`.

## CLI Flow

```mermaid
flowchart TD
    A[python -m rag_system.qa_cli] --> B[_build_parser]
    B --> C[parse args]
    C --> D[logging.basicConfig]
    D --> E[HybridRetriever.load]
    E --> F{question supplied?}
    F -->|yes| G[answer_question]
    F -->|no| H[interactive loop]
    H --> G
    G --> I[print answer]
```

# Function Call Flow

## Table of Contents
- [QA Call Graph](#qa-call-graph)
- [Ingestion Call Graph](#ingestion-call-graph)
- [Bulk Call Graph](#bulk-call-graph)

## QA Call Graph

```mermaid
flowchart TD
    main[qa_cli.main] --> load[HybridRetriever.load]
    main --> answer[answer_question]
    answer --> parse[parse_query]
    answer --> build[build_retrieval_query]
    answer --> ret[HybridRetriever.retrieve]
    ret --> mf[_metadata_filter]
    ret --> ds[_dense_scores]
    ret --> ls[_lexical_scores]
    ret --> rr[CrossEncoderReranker.rerank]
    ret --> ctx[format_context]
    answer --> llm[answer_with_groq]
```

## Ingestion Call Graph

```mermaid
flowchart TD
    main[DataIngestion.main] --> ingest[ingest]
    ingest --> pdf[load_pdf]
    ingest --> meta[extract_document_metadata]
    ingest --> chunk[chunk_pages]
    chunk --> clean[clean_pdf_pages]
    chunk --> units[_units_to_chunks]
    chunk --> mchunks[_build_metadata_chunks]
    ingest --> norm[normalize_record]
    ingest --> wire[wire_neighbors]
    ingest --> emb[PubMedEmbedder.encode_chunks]
```

## Bulk Call Graph

```mermaid
flowchart TD
    run[run_bulk_inference] --> loadq[load_questions]
    run --> retr[HybridRetriever.load]
    run --> proc[process_question]
    proc --> parse[parse_query]
    proc --> retrieve[HybridRetriever.retrieve]
    proc --> groq[answer_with_groq]
    run --> save[save_results]
```

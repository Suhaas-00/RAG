# Document Ingestion

## Table of Contents
- [Entry Points](#entry-points)
- [PDF Discovery](#pdf-discovery)
- [Per-PDF Processing](#per-pdf-processing)
- [Quality Filtering](#quality-filtering)
- [Embedding and Indexing](#embedding-and-indexing)
- [Outputs](#outputs)
- [Failure Handling](#failure-handling)

## Entry Points

The main implementation is top-level `DataIngestion.py`. `rag_system.ingestion.DataIngestion` re-exports `ingest`, `load_pdf`, and `main`. `rag_system.ingestion.ingest_cli` delegates to that wrapper.

## PDF Discovery

`ingest` validates `pdf_dir`, then discovers PDFs with `sorted(pdf_dir.rglob("*.pdf"))`. If no PDFs are found, it raises `ValueError`.

## Per-PDF Processing

For each PDF, ingestion assigns a `paper_id` using `build_paper_map`, loads pages with `load_pdf`, concatenates text for metadata extraction, builds chunks with `chunk_pages`, normalizes each record with `normalize_record`, filters weak chunks, wires neighbors, and appends records to the corpus list.

## Quality Filtering

`_is_quality_chunk` always keeps metadata chunks. Content chunks must satisfy `token_count(text) >= min_chunk_tokens` and alphabetic-character ratio >= `0.45`.

## Embedding and Indexing

`PubMedEmbedder(settings.model_name, use_prefixes=True)` embeds all records. FAISS uses `IndexFlatIP(embedder.dimension)`. TF-IDF is fitted with `ngram_range=(1,2)`, `min_df=1`, `sublinear_tf=True`, `norm="l2"`, and word analyzer.

## Outputs

| Output | Path under output dir | Description |
| --- | --- | --- |
| FAISS index | `faiss_index/vectors.index` | Active vector index. |
| Payload | `faiss_index/metadata.pkl` | Authoritative retrieval payload. |
| Chunk manifest | `chunks_manifest.json` | Human-readable chunk records. |
| Document metadata | `doc_metadata_index.json` | Metadata keyed by source, stem, and paper ID. |

## Failure Handling

Missing `pdf_dir`, no PDFs, and no usable chunks are fatal. Individual unreadable PDFs are skipped with warnings. CLI `main` catches `FileNotFoundError` and `ValueError` and exits with status 1.

# Function Reference

## Table of Contents
- [Root Ingestion](#root-ingestion)
- [Cleaning and Preprocessing](#cleaning-and-preprocessing)
- [Chunking and Embedding](#chunking-and-embedding)
- [Metadata and Query Parsing](#metadata-and-query-parsing)
- [Active Retrieval and Reranking](#active-retrieval-and-reranking)
- [LLM and CLI](#llm-and-cli)
- [Batch Runner](#batch-runner)
- [Alternate Retrieval Package](#alternate-retrieval-package)

## Root Ingestion

| Function | Inputs | Output | Algorithm and side effects | Complexity |
| --- | --- | --- | --- | --- |
| `DataIngestion._existing_or_default` | primary path, legacy path | string path | Chooses organized path unless only legacy exists. | O(1) |
| `DataIngestion.load_pdf` | `Path` | `list[PageText]` | Opens PDF with `PdfReader`, extracts every page. Reads disk. | O(pages) plus PDF extraction |
| `DataIngestion._is_quality_chunk` | chunk dict, min tokens | bool | Keeps metadata; filters content by token count and alphabetic ratio. | O(text length) |
| `DataIngestion.ingest` | pdf dir, output dir, optional settings | chunk count int | Full indexing pipeline. Writes FAISS, pickle, JSON. | O(PDF text + N embeddings + TF-IDF) |
| `DataIngestion._build_parser` | none | `ArgumentParser` | Defines ingestion CLI args. | O(1) |
| `DataIngestion.main` | CLI args | none | Parses args, configures logging, calls `ingest`, exits on fatal errors. | Pipeline-dependent |

## Cleaning and Preprocessing

| Function | Inputs | Output | Algorithm and side effects | Complexity |
| --- | --- | --- | --- | --- |
| `canonical_section` | section string | canonical string | Normalizes whitespace/lowercase and maps aliases. | O(length) |
| `detect_section_heading` | line | section or None | Regex match against known section headings. | O(length) |
| `normalize_text` | text | normalized text | NFKC, soft-hyphen removal, CR normalization, hyphen-line join. | O(length) |
| `_line_quality` | line, page | bool | Applies PDF noise heuristics. | O(length) |
| `remove_repeated_headers_footers` | page text list | cleaned page text list | Counts repeated first/last page lines and removes them. | O(total lines) |
| `clean_pdf_pages` | page objects | `list[CleanPage]` | Converts page objects, removes repeated headers, filters lines, preserves headings. | O(total text) |
| `clean_chunk_text` | text | cleaned string | Removes citations and figure/table sentences. | O(length) |
| `clean_text` | text, lowercase flag | cleaned string | Shared NFKC/citation/whitespace cleaning. | O(length) |
| `normalize_for_embedding` | text | lowercased cleaned string | Calls `clean_text(lowercase=True)`. | O(length) |
| `token_count` | text | int | Regex token estimate. | O(length) |
| `keyword_terms` | text | set[str] | Normalizes then extracts keyword tokens. | O(length) |
| `extract_metadata_from_text` | full text | dict | Regex metadata extraction from head/full text. | O(length) |

## Chunking and Embedding

| Function/method | Inputs | Output | Algorithm and side effects | Complexity |
| --- | --- | --- | --- | --- |
| `PageText.__post_init__` | self | none | Validates page number and normalizes empty text. | O(1) |
| `_canonical_section` | raw section | string | Local alias map. | O(length) |
| `_chunk_id` | source, ordinal, text | 20-char hex id | SHA-256 over source, ordinal, and text prefix. | O(1) bounded prefix |
| `_split_into_sentences` | text | list[str] | Regex split on sentence/paragraph boundaries. | O(length) |
| `_pages_to_units` | pages | unit list | Cleans pages, detects sections, emits sentence units. | O(total text) |
| `_units_to_chunks` | units, source, size, overlap, ordinal | chunk list | Packs section-homogeneous sentence windows. | O(units * overlap scan) |
| `_truncate_to_tokens` | text, max tokens | string | Adds words until token budget reached. | O(words) |
| `_build_metadata_chunks` | pages, source, metadata, ordinal | chunk list | Creates structured field and abstract chunks. | O(metadata + first pages) |
| `chunk_pages` | pages, source, params, metadata | chunk list | Validates, creates content and metadata chunks. | O(total text) |
| `wire_neighbors` | chunks | none | Mutates content chunk neighbor IDs and ordinals. | O(chunks) |
| `PubMedEmbedder.__init__` | model name, device, prefixes | instance | Loads SentenceTransformer. | Model-load dependent |
| `PubMedEmbedder.dimension` | self | int | Reads embedding dimension from model API. | O(1) |
| `PubMedEmbedder.encode` | texts, batch size, chunk types | ndarray | Normalizes, prefixes, encodes, L2-normalizes. | Model inference |
| `PubMedEmbedder.encode_chunks` | chunk dicts | ndarray | Extracts texts and chunk types, calls `encode`. | Model inference |
| `PubMedEmbedder.encode_query` | query | ndarray shape `(1,d)` | Encodes one unprefixed query. | Model inference |

## Metadata and Query Parsing

| Function | Inputs | Output | Algorithm and side effects | Complexity |
| --- | --- | --- | --- | --- |
| `build_paper_map` | PDF paths | dict | Assigns sorted `paper N` labels. | O(n log n) |
| `paper_id_for_source` | source, paper map | label/stem | Case-insensitive filename lookup. | O(n) |
| `extract_document_metadata` | text, source, paper ID | dict | Regex DOI/PMID/year/disease/gene/study extraction. | O(text length) |
| `normalize_record` | record, paper ID, chunk ID | dict | Standardizes chunk schema and aliases. | O(metadata size) |
| `_detect_section` | query | section/None | Alias scan with regex word boundaries. | O(aliases * query length) |
| `_resolve_paper` | query, paper map | label/source tuple | Matches paper references, filenames, or numeric stems. | O(papers) |
| `parse_query` | query, paper map | `QueryIntent` | Regex intent detection and filter construction. | O(query + papers) |
| `build_retrieval_query` | intent | string | Removes references and stop command words. | O(query length) |

## Active Retrieval and Reranking

| Function/method | Inputs | Output | Algorithm and side effects | Complexity |
| --- | --- | --- | --- | --- |
| `HybridRetriever.__init__` | index, payload, embedder, alpha | instance | Stores payload and creates reranker. | O(1) plus reranker load |
| `HybridRetriever.load` | index dir, alpha | retriever | Reads FAISS and pickle, loads embedder. | O(index/payload load) |
| `HybridRetriever.retrieve` | query, intent, k, alpha | `RetrievalResult` | Filters, dense search, lexical scoring, reranking, context formatting. | O(N + C*d + lexical + rerank) |
| `_metadata_filter` | filters | list[int] | Scans records and applies active filters. | O(N) |
| `_dense_scores` | query, candidate indices, k | dict | Embeds query, reconstructs vectors, temporary FAISS search. | O(C*d) plus embedding |
| `_lexical_scores` | query, candidate indices | dict | Candidate-local BM25-style scoring. | O(candidate tokens) |
| `list_papers` | none | list tuples | Returns payload paper map or derives from sources. | O(papers) |
| `format_context` | chunks | string | Builds context headers and joins chunks. | O(total text) |
| `_tokens` | text | list[str] | Normalized regex tokenization. | O(length) |
| `CrossEncoderReranker.__init__` | model name | instance | Attempts CrossEncoder load; catches failures. | Model-load dependent |
| `CrossEncoderReranker.rerank` | query, candidates, top_k | list[dict] | Predicts pair scores or uses fallback scores, sorts. | O(C log C) plus model inference |
| `_confidence` | rerank score, hybrid score | float | Clips and blends score components. | O(1) |

## LLM and CLI

| Function | Inputs | Output | Algorithm and side effects | Complexity |
| --- | --- | --- | --- | --- |
| `_build_user_prompt` | question, context | string | Formats retrieved context and instructions. | O(context length) |
| `load_groq_api_key` | optional start path | key or None | Searches upward for `.env`, loads it, reads env. Prints diagnostics on missing key. | O(parent dirs) |
| `answer_with_groq` | question, context, model | string | Loads key, builds prompt, calls Groq if key exists. | Network/API dependent |
| `_paper_map` | retriever | dict | Converts list of paper tuples to dict. | O(papers) |
| `_metadata_answer` | intent, retriever | string | Reads `doc_metadata_index` and formats field/all metadata. | O(candidate keys) |
| `_list_papers_answer` | retriever | string | Formats paper list. | O(papers) |
| `_format_direct_section` | retrieval result | string | Formats chunks without LLM. | O(total text) |
| `answer_question` | question, retriever, model, params | string | Main QA orchestration. | Retrieval/LLM dependent |
| `qa_cli._build_parser` | none | parser | Defines QA CLI args. | O(1) |
| `qa_cli.main` | CLI args | none | Loads retriever and runs one-shot or interactive mode. | Runtime dependent |

## Batch Runner

| Function | Inputs | Output | Algorithm and side effects | Complexity |
| --- | --- | --- | --- | --- |
| `_safe_get` | object, keys, default | value | Returns first non-None dict value. | O(keys) |
| `_extract_chunk_metadata` | chunk, rank | dict | Maps chunk fields to Excel columns. | O(1) |
| `load_questions` | path | DataFrame | Reads Excel, validates columns. | O(rows) |
| `build_summary_rows` | detail rows | summary rows | Groups by `SNo`, joins chunk IDs. | O(rows) |
| `save_results` | rows, path | none | Writes Excel detailed and summary sheets. | O(rows) |
| `process_question` | SNo, question, retriever, paper map | row list | Parses, retrieves, calls LLM, builds chunk rows. | Retrieval/LLM dependent |
| `run_bulk_inference` | input/output/model/k/alpha | none | Loads index once, processes workbook rows, writes report. | O(questions * retrieval/LLM) |

## Alternate Retrieval Package

| Function/method | Role |
| --- | --- |
| `RAGRetriever.load` | Loads `vectors.index` and `metadata.pkl`, validates schema. |
| `RAGRetriever._apply_metadata_filters` | Supports source, section, disease, gene, year, chunk type filters. |
| `RAGRetriever.retrieve` | Uses FAISS and stored TF-IDF matrix to return context and chunks. |
| `RAGRetriever.retrieve_by_metadata` | Returns metadata records without dense search. |
| `RAGRetriever._faiss_vector_at` | Reconstructs one FAISS vector. |
| `retrieval.hybrid_search.is_noisy` | Detects table/figure/reference/numeric-heavy text. |
| `retrieval.hybrid_search.keyword_score` | Jaccard-style keyword overlap. |
| `retrieval.hybrid_search._source_positions` | Filters clean candidate positions. |
| `retrieval.hybrid_search._faiss_subset_search` | Temporary FAISS subset search. |
| `retrieval.hybrid_search.hybrid_search` | Alternate dense + keyword candidate retrieval. |
| `retrieval.reranker.section_boost` | Section prior lookup. |
| `retrieval.reranker.rerank` | Alternate deterministic section-aware reranking. |

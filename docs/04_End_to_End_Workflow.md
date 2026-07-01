# End-to-End Workflow

## Table of Contents
- [Command](#command)
- [Runtime Trace](#runtime-trace)
- [Answer Paths](#answer-paths)
- [Batch Workflow](#batch-workflow)
- [Ingestion Workflow](#ingestion-workflow)

## Command

```powershell
python -m rag_system.qa_cli "question text"
```

Omit the positional question to enter interactive mode.

## Runtime Trace

1. `qa_cli.main` builds the argument parser.
2. Defaults are `--index-dir str(Settings().index_dir)`, `--model llama-3.1-8b-instant`, `--top-k 5`, `--alpha 0.55`, and `--log-level WARNING`.
3. Logging is initialized.
4. `HybridRetriever.load` reads `vectors.index` or legacy `index.faiss`, and requires `metadata.pkl`.
5. The payload is unpickled and `PubMedEmbedder` loads the payload model name.
6. `HybridRetriever` constructs `CrossEncoderReranker`.
7. `answer_question` parses the query with `parse_query` and rewrites it with `build_retrieval_query`.
8. `list_papers` and `metadata_query` intents are handled without LLM retrieval generation.
9. Other intents call `HybridRetriever.retrieve`.
10. `_metadata_filter` restricts candidates by source, paper ID, section, and chunk type when filters exist.
11. `_dense_scores` embeds the query, reconstructs candidate vectors, builds a temporary FAISS index, and searches it.
12. `_lexical_scores` computes query-local BM25-style lexical scores.
13. Scores are combined as `alpha * dense + (1 - alpha) * lexical`.
14. `CrossEncoderReranker.rerank` reranks if the CrossEncoder loaded; otherwise it uses hybrid scores.
15. `format_context` creates the context header and text blocks.
16. `answer_with_groq` loads `GROQ_API_KEY` from the process environment or `.env`.
17. With a key, Groq chat completion is called using temperature `0.1`.
18. Without a key, the LLM wrapper returns retrieved context or fallback.
19. The final string is printed.

## Answer Paths

| Intent/path | Function | LLM used? |
| --- | --- | --- |
| List papers | `_list_papers_answer` | No |
| Metadata query | `_metadata_answer` | No |
| Paper/section lookup with no key | `_format_direct_section` | No |
| Semantic QA | `answer_with_groq` | Yes if key exists |

## Batch Workflow

`qa_bulkload.run_bulk_inference` loads `datasets/questions/questions.xlsx`, loads the index once, builds `paper_map` once, processes each row with the same parser/retriever/LLM modules, and writes `outputs/reports/rag_results.xlsx`.

## Ingestion Workflow

`DataIngestion.main` parses ingestion flags, creates `Settings`, calls `ingest`, and prints the indexed chunk count. `ingest` writes `vectors.index`, `metadata.pkl`, `chunks_manifest.json`, and `doc_metadata_index.json`.

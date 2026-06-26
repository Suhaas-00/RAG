"""Backward-compatible CLI for the production medical RAG system."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag_system.hybrid_retriever import HybridRetriever, RetrievalResult
from rag_system.llm import FALLBACK, answer_with_groq, load_groq_api_key
from rag_system.query_parser import QueryIntent, build_retrieval_query, parse_query
from rag_system.utils import debug_dump
from rag_system.utils.config import Settings

logger = logging.getLogger(__name__)


def _paper_map(retriever: HybridRetriever) -> dict[str, str]:
    return {label: filename for label, filename in retriever.list_papers()}


def _metadata_answer(intent: QueryIntent, retriever: HybridRetriever) -> str:
    if not intent.paper_source and not intent.paper_label:
        return FALLBACK
    keys = [intent.paper_source, Path(intent.paper_source or "").stem, intent.paper_label]
    meta = None
    for key in keys:
        if key and key in retriever.payload.get("doc_metadata_index", {}):
            meta = retriever.payload["doc_metadata_index"][key]
            break
    if not meta:
        return FALLBACK
    field = intent.metadata_field
    if field and meta.get(field):
        return f"{field.upper()}: {meta[field]} (Source: {meta.get('source')}; Section: metadata; Page: 1; Confidence: 1.0)"
    lines = [f"Metadata for {meta.get('paper_id', intent.paper_label)} / {meta.get('source')}:"]
    for key in ("doi", "pmid", "year", "journal"):
        if meta.get(key):
            lines.append(f"- {key.upper()}: {meta[key]}")
    for key, label in (("diseases", "Diseases"), ("genes", "Genes"), ("study_designs", "Study designs")):
        if meta.get(key):
            lines.append(f"- {label}: {', '.join(meta[key])}")
    lines.append(f"(Source: {meta.get('source')}; Section: metadata; Page: 1; Confidence: 1.0)")
    return "\n".join(lines)


def _list_papers_answer(retriever: HybridRetriever) -> str:
    rows = retriever.list_papers()
    if not rows:
        return FALLBACK
    return "\n".join(f"{label}: {filename}" for label, filename in rows)


def _format_direct_section(result: RetrievalResult) -> str:
    if not result.chunks:
        return FALLBACK
    lines: list[str] = []
    for chunk in result.chunks:
        page = chunk.get("page", chunk.get("page_number", "?"))
        lines.append(
            f"Source: {chunk.get('source')} | Paper: {chunk.get('paper_id')} | "
            f"Section: {chunk.get('section')} | Page: {page} | "
            f"Confidence: {chunk.get('confidence', '?')}\n{chunk.get('text', '')}"
        )
    return "\n\n".join(lines)


def answer_question(
    question: str,
    retriever,
    model: str,
    *,
    top_k: int = 5,
    alpha: float = 0.55,
    verbose: bool = False,
) -> str:
    """Answer a question while preserving the old public function signature."""

    if not isinstance(retriever, HybridRetriever):
        # Compatibility with old imports that still instantiate RAGRetriever.
        if hasattr(retriever, "index") and hasattr(retriever, "payload") and hasattr(retriever, "embedder"):
            retriever = HybridRetriever(retriever.index, retriever.payload, retriever.embedder, alpha=alpha)
        else:
            raise TypeError("retriever must be HybridRetriever-compatible")

    intent = parse_query(question, _paper_map(retriever))
    retrieval_query = build_retrieval_query(intent)

    debug_dump(
        "query",
        {
            "detected_intent": intent.intent,
            "detected_paper": intent.paper_source or intent.paper_label,
            "detected_section": intent.section,
            "metadata_filters": intent.filters,
            "retrieval_query": retrieval_query,
        },
        verbose,
    )

    if intent.intent == "list_papers":
        return _list_papers_answer(retriever)
    if intent.intent == "metadata_query":
        return _metadata_answer(intent, retriever)

    result = retriever.retrieve(retrieval_query, intent, top_k=top_k, alpha=alpha)
    debug_dump("retrieval", result.debug, verbose)
    if not result.chunks:
        return FALLBACK

    if intent.intent in {"paper_lookup", "section_lookup"} and not load_groq_api_key(Path.cwd()):
        return _format_direct_section(result)

    return answer_with_groq(question, result.context, model=model)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa_cli",
        description="Query the medical RAG index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", nargs="?", help="Question to answer. Omit for interactive mode.")
    parser.add_argument("--index-dir", default=str(Settings().index_dir), metavar="PATH")
    parser.add_argument("--model", default="llama-3.1-8b-instant", metavar="NAME")
    parser.add_argument("--top-k", type=int, default=5, metavar="INT")
    parser.add_argument("--alpha", type=float, default=0.55, metavar="FLOAT")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    try:
        retriever = HybridRetriever.load(args.index_dir, alpha=args.alpha)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[ERROR] Could not load index from '{args.index_dir}': {exc}")

    if args.question:
        print(answer_question(args.question, retriever, args.model, top_k=args.top_k, alpha=args.alpha, verbose=args.verbose))
        return

    print("Medical RAG interactive mode (type 'quit' or Ctrl-D to exit)\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print(answer_question(question, retriever, args.model, top_k=args.top_k, alpha=args.alpha, verbose=args.verbose))


if __name__ == "__main__":
    main()

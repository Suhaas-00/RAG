"""Interactive question-answering CLI backed by the synchronized RAG pipeline.

Usage
-----
Single-shot:
    python -m rag_system.qa_cli "What disease is described in 12345678.pdf?"

Interactive REPL:
    python -m rag_system.qa_cli

Flags:
    --index-dir  PATH   Override the default FAISS index directory.
    --model      NAME   Groq model identifier (default: llama-3.1-8b-instant).
    --top-k      INT    Number of chunks to surface (default: 3).
    --verbose           Emit retrieved chunks and scores before the answer.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from rag_system.retrieval.retriever import RAGRetriever
from rag_system.utils.config import Settings

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

FALLBACK: str = "Not found in the document"

SECTIONS: tuple[str, ...] = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
)

_SYSTEM_PROMPT: str = (
    "You are a strictly grounded medical-document assistant. "
    "Answer only from the supplied context. "
    "If the context does not contain the answer, reply exactly: "
    f'"{FALLBACK}". '
    "Never speculate or draw on outside knowledge."
)

_DISEASE_PATTERN = re.compile(
    r"\b(?:lung cancer|breast cancer|prostate cancer|colorectal cancer|skin cancer|"
    r"adenocarcinoma|squamous cell carcinoma|carcinoma|lymphoma|leukemia|melanoma|"
    r"diabetes(?: mellitus)?|hypertension|stroke|asthma|covid-19|tuberculosis|"
    r"alzheimer(?:'s)? disease|parkinson(?:'s)? disease|cardiovascular disease|"
    r"heart failure|chronic kidney disease|obesity|arthritis|hepatitis|sepsis|"
    r"pneumonia)\b",
    re.IGNORECASE,
)

_TYPO_MAP: dict[str, str] = {
    "wha": "what",
    "wat": "what",
    "us": "is",
    "desease": "disease",
    "disese": "disease",
    "retreiver": "retriever",
    "retreival": "retrieval",
}

_STOP_WORDS: frozenset[str] = frozenset(
    {"what", "which", "is", "are", "the", "a", "an", "in", "of", "pdf", "paper", "file", "tell", "me", "about"}
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query classification helpers
# ---------------------------------------------------------------------------


def extract_paper_id(query: Optional[str]) -> Optional[str]:
    """Return the first run of 6+ digits found in *query* (PubMed ID or similar)."""
    match = re.search(r"(\d{6,})", query or "")
    return match.group(1) if match else None


def extract_section(query: Optional[str]) -> Optional[str]:
    """Return the first canonical section name found in *query*, or ``None``."""
    lowered = (query or "").lower()
    for section in SECTIONS:
        if section in lowered:
            return section
    return None


def classify_query(query: Optional[str]) -> str:
    """Route a query to one of four handler categories.

    Returns
    -------
    "list_papers"         – caller wants an inventory of indexed PDFs.
    "disease_query"       – caller asks which diseases appear in the corpus.
    "paper_section_query" – caller targets a specific section of a specific paper.
    "normal_query"        – generic; fall through to vector retrieval.
    """
    lowered = (query or "").lower()
    if "list" in lowered and "paper" in lowered:
        return "list_papers"
    if "disease" in lowered:
        return "disease_query"
    if extract_paper_id(lowered) and extract_section(lowered):
        return "paper_section_query"
    return "normal_query"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(text: Optional[str]) -> str:
    """Repair common PDF tokenisation artifacts while preserving clinical meaning.

    Transformations applied
    -----------------------
    * Soft-hyphen word joins (``multi- plexed`` → ``multiplexed``).
    * Known run-together compound (``oncogeneaddicted`` → ``oncogene-addicted``).
    * Collapse whitespace runs.
    * Split run-together CamelCase that originates from PDF extraction (``FooBar`` → ``Foo Bar``).
    """
    if not text:
        return ""
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"\boncogeneaddicted\b", "oncogene-addicted", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.strip()


def is_noisy(text: Optional[str]) -> bool:
    """Return ``True`` for chunks that are tables, figures, or OCR debris."""
    lowered = (text or "").lower()
    if any(marker in lowered for marker in ("table", "figure", "downloaded from", "references")):
        return True
    digit_ratio = sum(ch.isdigit() for ch in lowered) / max(len(lowered), 1)
    return digit_ratio > 0.30


# ---------------------------------------------------------------------------
# Disease extraction
# ---------------------------------------------------------------------------


def extract_diseases(records: list[dict]) -> list[str]:
    """Aggregate disease entities from metadata, supplemented by regex over text.

    Strategy
    --------
    1. Prefer structured ``entities`` list (values prefixed with ``"disease:"``).
    2. Fall back to regex over chunk text for records that are not noisy.
    """
    diseases: set[str] = set()
    for record in records:
        # Prefer pre-extracted structured entities.
        for entity in record.get("entities", []):
            if isinstance(entity, str) and entity.lower().startswith("disease:"):
                diseases.add(entity.split(":", 1)[1].strip().lower())
        # Supplement with regex over raw text.
        text = record.get("text", "")
        if not is_noisy(text):
            for match in _DISEASE_PATTERN.finditer(text):
                diseases.add(match.group(0).lower())
    return sorted(diseases)


# ---------------------------------------------------------------------------
# Structured metadata answers (no vector retrieval needed)
# ---------------------------------------------------------------------------


def list_papers(retriever: RAGRetriever) -> str:
    """Return a formatted inventory of all PDFs indexed in the payload."""
    sources = {
        Path(str(record.get("source", ""))).stem
        for record in retriever.payload.get("records", [])
        if record.get("source")
    }
    if not sources:
        return FALLBACK
    lines = "\n".join(f"  • {stem}" for stem in sorted(sources))
    return f"📄 Papers in the index ({len(sources)}):\n{lines}"


def disease_response(retriever: RAGRetriever, paper_id: Optional[str] = None) -> str:
    """Return a formatted list of diseases, optionally scoped to *paper_id*."""
    records: list[dict] = retriever.payload.get("records", [])
    if paper_id:
        records = [
            r for r in records
            if Path(str(r.get("source", ""))).stem.lower() == paper_id.lower()
        ]
    diseases = extract_diseases(records)
    if not diseases:
        return FALLBACK
    lines = "\n".join(f"  - {disease}" for disease in diseases)
    return f"🧬 Diseases mentioned{f' in {paper_id}' if paper_id else ''}:\n{lines}"


# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------


def preprocess_query(query: str) -> dict[str, Optional[str]]:
    """Normalise a raw query string and extract a PDF source constraint.

    Returns a dict with keys:
    ``clean_query``   – stop-word-stripped, typo-corrected query text.
    ``source_filter`` – PDF filename to pass to the retriever (may be ``None``).
    """
    lowered = (query or "").lower().strip()
    paper_id = extract_paper_id(lowered)

    # Prefer an explicit *.pdf* reference; fall back to a bare numeric ID.
    source_match = re.search(r"\b([a-z0-9_.\-]+\.pdf)\b", lowered, re.IGNORECASE)
    if source_match:
        source_filter: Optional[str] = source_match.group(1)
        lowered = lowered[: source_match.start()] + " " + lowered[source_match.end():]
    else:
        source_filter = f"{paper_id}.pdf" if paper_id else None

    # Conservative typo correction on individual tokens.
    words = re.findall(r"[a-z0-9\-]+", lowered)
    words = [_TYPO_MAP.get(w, w) for w in words]

    # Disease questions gain semantic coverage from section headings.
    if "disease" in words and any(w in words for w in {"describe", "described", "study"}):
        clean_query = "disease described study abstract introduction"
    else:
        clean_query = " ".join(w for w in words if w not in _STOP_WORDS)

    return {"clean_query": clean_query.strip() or lowered, "source_filter": source_filter}


# ---------------------------------------------------------------------------
# Core answer function
# ---------------------------------------------------------------------------


def answer_question(
    question: str,
    retriever: RAGRetriever,
    model: str,
    *,
    top_k: int = 3,
    verbose: bool = False,
) -> str:
    """Retrieve grounded context and generate an answer via Groq.

    Parameters
    ----------
    question:  Raw user question string.
    retriever: Loaded :class:`RAGRetriever` instance.
    model:     Groq model identifier.
    top_k:     Maximum number of retrieved chunks to feed into the prompt.
    verbose:   When ``True``, emit chunk metadata to *stderr* before the answer.

    Returns
    -------
    A grounded answer string, or :data:`FALLBACK` when no evidence is found.
    """
    question = (question or "").strip()
    if not question:
        logger.warning("Empty question received; returning fallback.")
        return FALLBACK

    query_type = classify_query(question)
    paper_id = extract_paper_id(question)
    section = extract_section(question)

    logger.debug("query_type=%s paper_id=%s section=%s", query_type, paper_id, section)

    # --- Structured metadata paths -------------------------------------------
    if query_type == "list_papers":
        return list_papers(retriever)
    if query_type == "disease_query":
        return disease_response(retriever, paper_id)

    # --- Vector retrieval path -----------------------------------------------
    processed = preprocess_query(question)
    retrieval_query = processed["clean_query"]

    if query_type == "paper_section_query" and section:
        retrieval_query = f"{section} of research paper"

    result = retriever.retrieve(
        retrieval_query,
        source_filter=processed["source_filter"],
        section_filter=section if query_type == "paper_section_query" else None,
        top_k=top_k,
    )

    if not result.context:
        logger.info("No relevant context found for question: %r", question)
        return FALLBACK

    if verbose:
        _log_chunks(result.chunks)

    # --- LLM synthesis -------------------------------------------------------
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Retrieval still works without a key; useful in integration tests.
        logger.warning("GROQ_API_KEY not set; returning raw retrieved context.")
        return result.context

    return _call_groq(question, result.context, model, api_key)


def _call_groq(question: str, context: str, model: str, api_key: str) -> str:
    """Send a grounded QA prompt to the Groq API and return the answer text."""
    try:
        from groq import Groq  # Late import: optional dependency.
    except ImportError as exc:
        raise RuntimeError(
            "The 'groq' package is required for LLM synthesis. "
            "Install it with:  pip install groq"
        ) from exc

    prompt = (
        f"Answer the question using ONLY the context below.\n"
        f'If the answer is absent, reply exactly: "{FALLBACK}"\n'
        f"Include the source and section for every factual claim. "
        f"Do not use outside knowledge.\n\n"
        f"CONTEXT\n{context}\n\n"
        f"QUESTION\n{question}"
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer or FALLBACK
    except Exception as exc:  # pragma: no cover
        logger.error("Groq API call failed: %s", exc)
        raise


def _log_chunks(chunks: list[dict]) -> None:
    """Write retrieved chunk metadata to *stderr* for debugging."""
    print("\n── Retrieved chunks ──", file=sys.stderr)
    for i, chunk in enumerate(chunks, 1):
        print(
            f"  [{i}] source={chunk.get('source')} "
            f"section={chunk.get('section')} "
            f"page={chunk.get('page_number')} "
            f"score={chunk.get('final_score', chunk.get('rerank_score', '?')):.4f}",
            file=sys.stderr,
        )
    print("─────────────────────\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa_cli",
        description="Query the synchronized medical RAG index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to answer. Omit for interactive REPL mode.",
    )
    parser.add_argument(
        "--index-dir",
        default=str(Settings().index_dir),
        metavar="PATH",
        help="Path to the FAISS index directory.",
    )
    parser.add_argument(
        "--model",
        default="llama-3.1-8b-instant",
        metavar="NAME",
        help="Groq model identifier.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="INT",
        help="Number of chunks to surface per query.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print retrieved chunk metadata before the answer.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return parser


def main() -> None:  # pragma: no cover
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        retriever = RAGRetriever.load(args.index_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"[ERROR] Could not load index from '{args.index_dir}': {exc}")

    if args.question:
        print(
            answer_question(
                args.question,
                retriever,
                args.model,
                top_k=args.top_k,
                verbose=args.verbose,
            )
        )
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("Medical RAG — interactive mode  (type 'quit' or Ctrl-D to exit)\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        try:
            print(
                answer_question(
                    question,
                    retriever,
                    args.model,
                    top_k=args.top_k,
                    verbose=args.verbose,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}", file=sys.stderr)
    print()


if __name__ == "__main__":
    main()
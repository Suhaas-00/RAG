"""
qa_bulkload.py
==============
Batch inference script for the Medical RAG system.

Reads questions from an input Excel file, runs retrieval + LLM generation
for every question, and writes one row per retrieved chunk to an output
Excel file.

Usage
-----
    python -m rag_system.qa_bulkload

All configurable values live at the top of this file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration – edit these values as needed
# ---------------------------------------------------------------------------

INPUT_EXCEL:  str   = "datasets/questions/questions.xlsx"  # Path to the input questions file
OUTPUT_EXCEL: str   = "outputs/reports/rag_results.xlsx"   # Path for the output results file
MODEL:        str   = "llama-3.1-8b-instant"  # Groq model identifier
TOP_K:        int   = 5                       # Number of chunks to retrieve per question
ALPHA:        float = 0.55                    # Hybrid retrieval weight (dense vs sparse)
ALLOW_GLOBAL_SEARCH: bool = False             # Explicitly allow unscoped corpus-wide retrieval

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output column order
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: list[str] = [
    "SNo",
    "Question",
    "Chunk Rank",
    "Chunk ID",
    "Paper ID",
    "Source File",
    "Section",
    "Page Number",
    "Confidence",
    "Chunk Text",
    "Final Response",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_get(obj: Any, *keys: str, default: Any = "") -> Any:
    """
    Safely retrieve a value from a dict using multiple candidate key names.
    Returns *default* when none of the keys are present or all values are None.

    Parameters
    ----------
    obj : dict or None
        The chunk dict to probe.
    *keys : str
        Candidate key names tried left-to-right.
    default : Any
        Fallback value (default: empty string).
    """
    if not isinstance(obj, dict):
        return default
    for key in keys:
        val = obj.get(key)
        if val is not None:
            return val
    return default


def _extract_chunk_metadata(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    """
    Extract output-column values from a single retrieved chunk dict.

    Based on qa_cli.py the chunks returned by HybridRetriever.retrieve()
    are plain dicts with these keys (all optional):
        text, source, paper_id, section, page, page_number, confidence

    Missing keys are silently replaced with an empty string.

    Parameters
    ----------
    chunk : dict
        A single chunk dict from RetrievalResult.chunks.
    rank : int
        1-based rank of this chunk within the current question's results.

    Returns
    -------
    dict
        Partial row covering all chunk-owned OUTPUT_COLUMNS.
    """
    # page is stored as either "page" or "page_number" (both used in qa_cli.py)
    page = _safe_get(chunk, "page", "page_number")

    return {
        "Chunk Rank" : rank,
        "Chunk ID"   : _safe_get(chunk, "chunk_id", "id"),
        "Paper ID"   : _safe_get(chunk, "paper_id", "doc_id"),
        "Source File": _safe_get(chunk, "source", "filename", "file_name"),
        "Section"    : _safe_get(chunk, "section", "header"),
        "Page Number": page,
        "Confidence" : _safe_get(chunk, "confidence", "score"),
        "Chunk Text" : _safe_get(chunk, "text", "content"),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_questions(path: str) -> pd.DataFrame:
    """
    Load the input Excel file and validate required columns.

    Parameters
    ----------
    path : str
        File-system path to the input .xlsx file.

    Returns
    -------
    pd.DataFrame
        DataFrame guaranteed to contain SNo and Question columns.

    Raises
    ------
    FileNotFoundError
        When path does not exist.
    ValueError
        When required columns are absent.
    """
    p = Path(path)
    if not p.exists() and path == INPUT_EXCEL and Path("questions.xlsx").exists():
        p = Path("questions.xlsx")
    if not p.exists():
        raise FileNotFoundError(f"Input Excel not found: {p.resolve()}")

    df = pd.read_excel(p, engine="openpyxl")
    logger.info("Loaded %d question(s) from '%s'.", len(df), p)

    missing = {"SNo", "Question"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Input Excel is missing required column(s): {missing}. "
            f"Found: {list(df.columns)}"
        )

    return df


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Collapse the per-chunk rows into one summary row per question.

    All Chunk IDs for a question are joined into a single cell separated
    by commas so the summary sheet has exactly one row per SNo.

    Parameters
    ----------
    rows : list[dict]
        The same per-chunk rows written to Sheet 1 (Detailed Results).

    Returns
    -------
    list[dict]
        One dict per unique SNo with keys:
        SNo, Question, Chunk IDs, Final Response.
    """
    seen: dict[Any, dict[str, Any]] = {}

    for row in rows:
        sno = row["SNo"]
        if sno not in seen:
            seen[sno] = {
                "SNo"           : sno,
                "Question"      : row["Question"],
                "Chunk IDs"     : [],
                "Final Response": row["Final Response"],
            }
        chunk_id = row.get("Chunk ID", "")
        if chunk_id:
            seen[sno]["Chunk IDs"].append(str(chunk_id))

    summary: list[dict[str, Any]] = []
    for entry in seen.values():
        entry["Chunk IDs"] = ", ".join(entry["Chunk IDs"]) if entry["Chunk IDs"] else ""
        summary.append(entry)

    return summary


SUMMARY_COLUMNS: list[str] = ["SNo", "Question", "Chunk IDs", "Final Response"]


def save_results(rows: list[dict[str, Any]], path: str) -> None:
    """
    Write results to an Excel file with two sheets.

    Sheet 1 - "Detailed Results"
        One row per retrieved chunk (all OUTPUT_COLUMNS).

    Sheet 2 - "Summary"
        One row per question with all Chunk IDs collapsed into a single
        comma-separated cell.
        Columns: SNo | Question | Chunk IDs | Final Response

    Parameters
    ----------
    rows : list[dict]
        List of flat dicts whose keys are a superset of OUTPUT_COLUMNS.
    path : str
        Destination .xlsx file path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detail_df  = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    summary_df = pd.DataFrame(build_summary_rows(rows), columns=SUMMARY_COLUMNS)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        detail_df.to_excel(writer,  sheet_name="Detailed Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary",          index=False)

    logger.info(
        "Wrote %d detail row(s) and %d summary row(s) to '%s'.",
        len(detail_df), len(summary_df), path,
    )


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------


def process_question(
    sno: Any,
    question: str,
    retriever: Any,
    paper_map: dict[str, str],
    *,
    top_k: int = TOP_K,
    model: str = MODEL,
    alpha: float = ALPHA,
    allow_global_search: bool = ALLOW_GLOBAL_SEARCH,
) -> list[dict[str, Any]]:
    """
    Run retrieval and LLM generation for one question.

    Mirrors the logic in qa_cli.py::answer_question() exactly, reusing the
    same imported functions without duplicating any logic.

    Parameters
    ----------
    sno : Any
        Serial number from the input sheet, preserved verbatim.
    question : str
        The natural-language question string.
    retriever : HybridRetriever
        A loaded HybridRetriever instance.
    paper_map : dict[str, str]
        Label -> filename mapping from retriever.list_papers(), built once
        and passed in to avoid rebuilding it for every question.
    top_k : int
        Maximum number of chunks to retrieve.
    model : str
        Groq model identifier for answer generation.
    alpha : float
        Hybrid retrieval blending weight.

    Returns
    -------
    list[dict]
        One dict per retrieved chunk. Empty list when no chunks are found.
    """
    # All imports match exact file locations from qa_cli.py
    from rag_system.query_parser import parse_query, build_retrieval_query
    from rag_system.llm import answer_with_groq

    # 1. Parse intent (pass paper_map so paper references resolve correctly)
    intent          = parse_query(question, paper_map)
    retrieval_query = build_retrieval_query(intent)
    logger.debug("SNo %s -> intent=%s  query=%s", sno, intent.intent, retrieval_query)

    # 2. Retrieve – signature from qa_cli.py:
    #    retriever.retrieve(retrieval_query, intent, top_k=top_k, alpha=alpha)
    result = retriever.retrieve(
        retrieval_query,
        intent,
        top_k=top_k,
        alpha=alpha,
        allow_global_search=allow_global_search,
    )

    if not result.chunks:
        logger.warning("No chunks retrieved for SNo %s: '%s'", sno, question)
        return []

    # 3. Generate the LLM answer once for this question.
    #    answer_with_groq(question, result.context, model=model) – from qa_cli.py
    final_response: str = answer_with_groq(question, result.context, model=model)

    # 4. One output row per chunk
    rows: list[dict[str, Any]] = []
    for rank, chunk in enumerate(result.chunks, start=1):
        chunk_data = _extract_chunk_metadata(chunk, rank)
        rows.append(
            {
                "SNo"           : sno,
                "Question"      : question,
                **chunk_data,
                "Final Response": final_response,
            }
        )

    logger.debug("SNo %s -> %d chunk row(s) produced.", sno, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_bulk_inference(
    input_excel:  str   = INPUT_EXCEL,
    output_excel: str   = OUTPUT_EXCEL,
    model:        str   = MODEL,
    top_k:        int   = TOP_K,
    alpha:        float = ALPHA,
    allow_global_search: bool = ALLOW_GLOBAL_SEARCH,
) -> None:
    """
    Main entry-point: load index once, iterate over questions, write output.

    Parameters
    ----------
    input_excel : str
        Path to the input .xlsx file with SNo / Question columns.
    output_excel : str
        Path for the output .xlsx results file.
    model : str
        Groq model identifier.
    top_k : int
        Number of chunks to retrieve per question.
    alpha : float
        Hybrid retrieval blending weight (dense=alpha, sparse=1-alpha).
    """
    # Exact imports taken from qa_cli.py – nothing guessed
    from rag_system.hybrid_retriever import HybridRetriever
    from rag_system.llm import load_groq_api_key
    from rag_system.utils.config import Settings          # qa_cli.py line: from rag_system.utils.config import Settings

    # -----------------------------------------------------------------------
    start_time = time.perf_counter()
    logger.info("=== Medical RAG - Bulk Inference ===")
    logger.info(
        "model=%s  top_k=%d  alpha=%.2f  allow_global_search=%s",
        model, top_k, alpha, allow_global_search,
    )

    # Load Groq API key (load_groq_api_key takes cwd as argument – from qa_cli.py)
    load_groq_api_key(Path.cwd())

    # Load questions from Excel
    questions_df = load_questions(input_excel)
    total: int = len(questions_df)
    logger.info("Questions to process: %d", total)

    # Load the RAG index ONCE and reuse for every question
    index_dir = str(Settings().index_dir)
    logger.info("Loading index from: %s", index_dir)
    retriever: Any = HybridRetriever.load(
        index_dir,
        alpha=alpha,
        allow_global_search=allow_global_search,
    )
    logger.info("Index loaded successfully.")

    # Build paper_map once (used by parse_query for paper-reference resolution)
    paper_map: dict[str, str] = {
        label: filename for label, filename in retriever.list_papers()
    }
    logger.info("Paper map: %d paper(s) registered.", len(paper_map))

    # -----------------------------------------------------------------------
    all_rows: list[dict[str, Any]] = []
    failures: int = 0

    for position, (_, row_data) in enumerate(questions_df.iterrows(), start=1):
        sno      = row_data["SNo"]
        question = str(row_data["Question"]).strip()

        print(f"Processing Question {position}/{total}", flush=True)
        logger.info("[%d/%d] SNo=%s | %s", position, total, sno, question)

        try:
            rows = process_question(
                sno=sno,
                question=question,
                retriever=retriever,
                paper_map=paper_map,
                top_k=top_k,
                model=model,
                alpha=alpha,
                allow_global_search=allow_global_search,
            )
            all_rows.extend(rows)
            logger.info("  -> %d chunk row(s) added for SNo %s.", len(rows), sno)

        except Exception as exc:  # noqa: BLE001 – one failure must not stop the batch
            failures += 1
            logger.error("  x SNo %s failed: %s", sno, exc, exc_info=True)
            # Placeholder row keeps the question visible in the output file
            all_rows.append(
                {col: "" for col in OUTPUT_COLUMNS}
                | {
                    "SNo"           : sno,
                    "Question"      : question,
                    "Final Response": f"ERROR: {exc}",
                }
            )

    # -----------------------------------------------------------------------
    # Persist results
    # -----------------------------------------------------------------------
    save_results(all_rows, output_excel)

    elapsed    = time.perf_counter() - start_time
    successful = total - failures

    summary = (
        f"\n{'=' * 52}\n"
        f"  Total processed : {total}\n"
        f"  Successful      : {successful}\n"
        f"  Failures        : {failures}\n"
        f"  Elapsed time    : {elapsed:.1f}s\n"
        f"{'=' * 52}"
    )
    print(summary)
    logger.info(
        "Done. total=%d  successful=%d  failures=%d  elapsed=%.1fs",
        total, successful, failures, elapsed,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_bulk_inference(
        input_excel  = INPUT_EXCEL,
        output_excel = OUTPUT_EXCEL,
        model        = MODEL,
        top_k        = TOP_K,
        alpha        = ALPHA,
        allow_global_search = ALLOW_GLOBAL_SEARCH,
    )

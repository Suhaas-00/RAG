"""Groq LLM wrapper with reliable .env loading, diagnostics, and
precision-oncology prompt for NSCLC / EGFR / MET biomarker RAG."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

FALLBACK = "Not found in the document"

# ---------------------------------------------------------------------------
# Citation format (referenced in both system and user prompts)
# ---------------------------------------------------------------------------
_CITATION_FORMAT = "(Source: <filename>; Section: <section>; Page: <n>; Confidence: <0.00–1.00>)"

# ---------------------------------------------------------------------------
# System prompt – precision oncology grounding
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    # Role
    "You are a grounded precision oncology RAG assistant specialising in "
    "biomarker-guided therapeutic decision-making for oncogene-driven lung "
    "cancers, especially Non-Small Cell Lung Cancer (NSCLC).\n\n"

    # Domain scope – tells the model which entities matter in this corpus
    "Your knowledge domain covers:\n"
    "  • Oncogenes & biomarkers: EGFR (exon 19 deletions, L858R, T790M) "
    "and MET (exon 14 skipping mutations, amplification, kinase domain mutations).\n"
    "  • Targeted therapies: Osimertinib, Erlotinib, Afatinib, Crizotinib, "
    "Tepotinib, Gefitinib, Savolitinib, Rociletinib.\n"
    "  • Clinical outcomes: drug sensitivity, objective response rate, "
    "acquired resistance mechanisms, progression-free survival.\n\n"

    # Hard grounding rules – numbered so the model treats them as a checklist
    "STRICT RULES — follow without exception:\n"
    "1. Answer ONLY from the supplied context. Do NOT use external or "
    "   pre-trained biomedical knowledge beyond what the context states.\n"
    f"2. Every factual claim MUST be followed by a citation in the exact form: {_CITATION_FORMAT}\n"
    "3. Do NOT invent, extrapolate, or infer beyond what the context explicitly states.\n"
    "4. If the context does not contain enough information to answer the question, "
    f"  respond with exactly: {FALLBACK}\n"
    "5. Never speculate about drug dosing, clinical recommendations, or "
    "   patient-specific treatment decisions.\n"
    "6. Use precise oncology terminology (e.g. 'EGFR exon 19 deletion' not "
    "   'EGFR mutation' when the context provides that specificity).\n\n"

    # Query-type awareness
    "Query types this system handles:\n"
    "  • Biomarker query   – mutations/variants/gene alterations\n"
    "  • Drug query        – mechanism, indication, resistance profile\n"
    "  • Outcome query     – sensitivity, response, resistance, PFS\n"
    "  • Disease query     – NSCLC subtypes, staging, patient population\n"
    "  • Relationship query– biomarker ↔ drug, drug ↔ outcome, gene ↔ disease\n\n"

    # Required output structure – three sections, every time
    "Structure every answer exactly as follows:\n\n"
    "ANSWER\n"
    "------\n"
    "<Concise, evidence-based response in 2–5 sentences using precise oncology "
    "terminology. Name specific biomarkers, drugs, or outcomes where relevant.>\n\n"
    "EVIDENCE\n"
    "--------\n"
    "List each supporting chunk on its own numbered line:\n"
    "  [1] <one-sentence summary of chunk content> {citation}\n"
    "  [2] ...\n\n"
    "CONFIDENCE SUMMARY\n"
    "------------------\n"
    "Overall confidence: <High / Moderate / Low>\n"
    "Reason: <one sentence explaining confidence based on chunk scores and "
    "relevance to the question>"
)


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------
def _build_user_prompt(question: str, context: str) -> str:
    """Inject retrieved context and question into the structured user turn."""
    return (
        "RETRIEVED CONTEXT\n"
        "=================\n"
        f"{context}\n"
        "=================\n\n"
        f"QUESTION: {question}\n\n"
        "Instructions:\n"
        "  1. Read every chunk in the context carefully before answering.\n"
        "  2. Identify which chunks are directly relevant to the question.\n"
        "  3. Synthesise a single coherent answer from those chunks only.\n"
        "  4. Cite every chunk you use with the required citation format.\n"
        "  5. Follow the ANSWER / EVIDENCE / CONFIDENCE SUMMARY structure exactly.\n"
        f"  6. If the context is insufficient, output exactly: {FALLBACK}"
    )


# ---------------------------------------------------------------------------
# .env / API key loader
# ---------------------------------------------------------------------------
def load_groq_api_key(start: Path | None = None) -> str | None:
    """
    Walk upward from *start* (default: cwd) looking for a .env file and
    load it into the environment.  Returns the GROQ_API_KEY value or None.
    """
    root = (start or Path.cwd()).resolve()
    candidates = [root / ".env", *root.parents]
    env_files = [
        path if path.name == ".env" else path / ".env"
        for path in candidates
    ]

    loaded_from = None
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=False)
            loaded_from = env_file
            break

    key = os.getenv("GROQ_API_KEY")
    if not key:
        print(
            f"[diagnostic] GROQ_API_KEY is missing. "
            f"Checked .env from {root} upward.",
            flush=True,
        )
        if loaded_from:
            print(f"[diagnostic] Loaded .env file: {loaded_from}", flush=True)

    return key


# ---------------------------------------------------------------------------
# Main LLM entry-point
# ---------------------------------------------------------------------------
def answer_with_groq(
    question: str,
    context: str,
    *,
    model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Generate a grounded, citation-rich answer using the Groq API.

    Uses a precision-oncology system prompt focused on NSCLC, EGFR/MET
    biomarkers, targeted therapies, and treatment outcomes.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    context : str
        Pre-formatted retrieval context string (result.context from
        HybridRetriever), containing chunk text + metadata.
    model : str
        Groq model identifier (default: llama-3.1-8b-instant).

    Returns
    -------
    str
        Structured model answer (ANSWER / EVIDENCE / CONFIDENCE SUMMARY)
        or FALLBACK when the API key is missing or context is insufficient.
    """
    api_key = load_groq_api_key()
    if not api_key:
        logger.warning("GROQ_API_KEY not found – returning context as fallback.")
        return context or FALLBACK

    from groq import Groq

    user_prompt = _build_user_prompt(question, context)

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,   # low = faithful to context, minimal hallucination
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    answer = (response.choices[0].message.content or "").strip()
    logger.debug("Groq response (%s): %s", model, answer[:120])
    return answer or FALLBACK
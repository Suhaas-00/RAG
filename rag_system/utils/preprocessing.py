"""Exactly-shared text normalisation for documents, queries, and keywords.

All text that enters the FAISS index or is used to query it **must** pass
through :func:`normalize_for_embedding`.  This contract guarantees that the
same tokenisation is applied at ingestion time and at query time, preventing
vocabulary mismatches that silently degrade retrieval quality.

Public API
----------
clean_text(text, *, lowercase)     – Remove PDF extraction noise.
normalize_for_embedding(text)      – Canonical preprocessing for FAISS.
token_count(text)                  – Conservative token budget estimator.
keyword_terms(text)                – Set of indexable keyword tokens.
extract_metadata_from_text(text)   – Heuristic metadata extraction from raw PDF text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for performance)
# ---------------------------------------------------------------------------

# C0/C1 control characters except tab (0x09), LF (0x0A), CR (0x0D).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Inline citation markers: [1], [2,3], [4–6], [7; 8], etc.
_CITATION = re.compile(r"\[(?:\d+[\s,;\-–]*)+\]")

# Any run of whitespace (including non-breaking spaces after NFKC).
_SPACE = re.compile(r"\s+")

# Three or more consecutive identical non-word, non-space characters.
_REPEATED_PUNCT = re.compile(r"([^\w\s])\1{2,}")

# PDF soft-hyphen word joins: "multi-\nplexed" → "multiplexed".
_PDF_HYPHEN = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")

# Keyword token pattern: lower-alpha-numeric, at least 2 characters.
_KEYWORD_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]{1,}")

# Token estimator: words and individual punctuation marks.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

# DOI pattern
_DOI_RE = re.compile(r"\b(10\.\d{4,}/[^\s]+)", re.IGNORECASE)

# PMID / PubMed pattern
_PMID_RE = re.compile(r"\b(?:PMID|PubMed[:\s]+ID)\s*[:\-]?\s*(\d{6,10})\b", re.IGNORECASE)

# Year pattern (plausible publication years)
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")

# Author line heuristic: "Surname AB, Surname CD" or comma-separated capitalised tokens
_AUTHOR_RE = re.compile(
    r"^(?:[A-Z][a-z]+ [A-Z]{1,3}(?:,\s*|\s+and\s+))+[A-Z][a-z]+ [A-Z]{1,3}",
)

# Journal name often follows author block or DOI
_JOURNAL_RE = re.compile(
    r"\b(?:N Engl J Med|Lancet|JAMA|BMJ|Nature|Science|Cell|"
    r"J Clin Oncol|Cancer Res|Cancer Discov|J Thorac Oncol|"
    r"Clin Cancer Res|Ann Oncol|Blood|Gut|Chest|Radiology|"
    r"The Oncologist|Oncologist)\b",
    re.IGNORECASE,
)

# Disease / condition terms used for metadata tagging
_DISEASE_PATTERN = re.compile(
    r"\b(?:lung cancer|breast cancer|prostate cancer|colorectal cancer|skin cancer|"
    r"adenocarcinoma|squamous cell carcinoma|carcinoma|lymphoma|leukemia|melanoma|"
    r"diabetes(?: mellitus)?|hypertension|stroke|asthma|covid-?19|tuberculosis|"
    r"alzheimer(?:'s)? disease|parkinson(?:'s)? disease|cardiovascular disease|"
    r"heart failure|chronic kidney disease|obesity|arthritis|hepatitis|sepsis|"
    r"pneumonia|nsclc|non.small.cell lung cancer|small.cell lung cancer|"
    r"mesothelioma|glioblastoma|glioma|renal cell carcinoma|"
    r"hepatocellular carcinoma|pancreatic cancer|ovarian cancer|"
    r"endometrial cancer|cervical cancer|thyroid cancer|"
    r"multiple myeloma|myelodysplastic syndrome|acute myeloid leukemia|"
    r"chronic lymphocytic leukemia|diffuse large b-cell lymphoma)\b",
    re.IGNORECASE,
)

# Gene / biomarker terms
_GENE_PATTERN = re.compile(
    r"\b(?:EGFR|ALK|ROS1|KRAS|BRAF|MET|HER2|ERBB2|PIK3CA|PTEN|TP53|RET|"
    r"NTRK[123]|FGFR[1-4]|CDK[46]|BRCA[12]|APC|VHL|NF1|NF2|"
    r"T790M|L858R|exon 14|MET exon)\b",
    re.IGNORECASE,
)

# Study design terms
_STUDY_DESIGN_RE = re.compile(
    r"\b(?:randomized|randomised|double.blind|placebo.controlled|"
    r"phase [I]{1,3}|phase [123]|cohort study|case.control|"
    r"meta.analysis|systematic review|retrospective|prospective|"
    r"clinical trial|open.label|crossover)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core text cleaner
# ---------------------------------------------------------------------------


def clean_text(text: Optional[str], *, lowercase: bool = False) -> str:
    """Remove PDF extraction noise without deleting medically meaningful content.

    Transformations applied (in order)
    -----------------------------------
    1. Unicode NFKC normalisation (collapses ligatures, normalises width, etc.).
    2. Remove soft hyphens (U+00AD).
    3. Normalise CR/CRLF to LF.
    4. Rejoin soft-hyphenated words split across PDF lines (``multi-\\nplexed``).
    5. Strip C0/C1 control characters (preserve tab, LF, CR).
    6. Remove inline citation markers (``[1]``, ``[2,3]``, …).
    7. Collapse runs of repeated punctuation (``...`` → ``.``).
    8. Collapse whitespace runs to a single space and strip leading/trailing.
    9. Optionally lower-case.

    Parameters
    ----------
    text:      Raw string from PDF extraction or user input. ``None`` returns ``""``.
    lowercase: When ``True``, return the result in lower case.

    Returns
    -------
    A clean, single-line string with medical punctuation preserved.
    """
    if not text:
        return ""

    # Step 1 – Unicode normalisation.
    text = unicodedata.normalize("NFKC", text)
    # Step 2 – soft hyphens are invisible but break tokenisation.
    text = text.replace("\u00ad", "")
    # Step 3 – normalise CR to LF.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Step 4 – rejoin PDF soft-hyphenated words.
    text = _PDF_HYPHEN.sub("", text)
    # Step 5 – control characters.
    text = _CONTROL.sub(" ", text)
    # Step 6 – citation markers pollute keyword scores.
    text = _CITATION.sub(" ", text)
    # Step 7 – repeated punctuation (``-----``, ``...`` etc.).
    text = _REPEATED_PUNCT.sub(r"\1", text)
    # Step 8 – whitespace normalisation.
    text = _SPACE.sub(" ", text).strip()
    # Step 9 – optional lower-casing.
    return text.lower() if lowercase else text


# ---------------------------------------------------------------------------
# Embedding normalisation (shared contract between ingestion and retrieval)
# ---------------------------------------------------------------------------


def normalize_for_embedding(text: Optional[str]) -> str:
    """Canonical preprocessing applied identically at ingestion and query time.

    This is the **single source of truth** for how text is prepared before
    it is encoded by the embedding model.  Any change here requires a full
    index rebuild.

    Parameters
    ----------
    text: Raw text string.

    Returns
    -------
    Cleaned, lower-cased string ready for :class:`PubMedEmbedder`.
    """
    return clean_text(text, lowercase=True)


# ---------------------------------------------------------------------------
# Token budget estimator
# ---------------------------------------------------------------------------


def token_count(text: Optional[str]) -> int:
    """Return a conservative token-count estimate for chunk and context budgets.

    The estimator counts word-like tokens and individual punctuation marks
    using a simple regex — it intentionally over-counts relative to
    sub-word tokenisers (BPE, WordPiece) so that budgets are never exceeded.

    Parameters
    ----------
    text: Any string. ``None`` or empty returns 0.

    Returns
    -------
    Non-negative integer token estimate.
    """
    return len(_TOKEN_PATTERN.findall(text or ""))


# ---------------------------------------------------------------------------
# Keyword terms
# ---------------------------------------------------------------------------


def keyword_terms(text: Optional[str]) -> set[str]:
    """Return a set of indexable keyword tokens from *text*.

    Tokens are lower-alpha-numeric strings of length ≥ 2, optionally
    containing internal hyphens (e.g. ``"covid-19"``, ``"non-small"``).
    Stop words are **not** removed here; callers that need filtered sets
    should subtract their own stop-word list.

    Parameters
    ----------
    text: Raw or partially cleaned string.

    Returns
    -------
    A ``set`` of normalised keyword strings.
    """
    return set(_KEYWORD_TOKEN.findall(normalize_for_embedding(text)))


# ---------------------------------------------------------------------------
# Metadata extraction from raw PDF text
# ---------------------------------------------------------------------------


def extract_metadata_from_text(full_text: str) -> dict:
    """Heuristically extract structured metadata from the raw text of a PDF.

    Inspects the first ~3 000 characters (title page / abstract area) for:
    - DOI
    - PubMed ID (PMID)
    - Publication year
    - Journal name
    - Disease / condition mentions
    - Gene / biomarker mentions
    - Study design keywords

    Returns
    -------
    A flat dict with string or list values, safe to serialise to JSON.
    All fields default to ``None`` / empty list when not found.
    """
    head = full_text[:3000]

    # DOI
    doi_match = _DOI_RE.search(head)
    doi = doi_match.group(1).rstrip(".,;)") if doi_match else None

    # PMID
    pmid_match = _PMID_RE.search(full_text)
    pmid = pmid_match.group(1) if pmid_match else None

    # Year – prefer the first plausible year in the header area
    year_matches = _YEAR_RE.findall(head)
    year = year_matches[0] if year_matches else None

    # Journal
    journal_match = _JOURNAL_RE.search(head)
    journal = journal_match.group(0) if journal_match else None

    # Diseases (deduplicated, sorted, from full text)
    diseases = sorted({m.group(0).lower() for m in _DISEASE_PATTERN.finditer(full_text)})

    # Genes (deduplicated, sorted, from full text)
    genes = sorted({m.group(0).upper() for m in _GENE_PATTERN.finditer(full_text)})

    # Study design
    study_designs = sorted({m.group(0).lower() for m in _STUDY_DESIGN_RE.finditer(head)})

    return {
        "doi": doi,
        "pmid": pmid,
        "year": year,
        "journal": journal,
        "diseases": diseases,
        "genes": genes,
        "study_designs": study_designs,
    }
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
    text:      Raw string from PDF extraction or user input.  ``None`` returns ``""``.
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
    text: Any string.  ``None`` or empty returns 0.

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
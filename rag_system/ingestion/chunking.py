"""Sentence- and section-aware PDF chunking with page provenance.

Overview
--------
The chunker converts a list of :class:`PageText` objects (one per PDF page)
into a flat list of chunk dicts that are ready for embedding.  Each chunk:

* Stays within a single paper section (abstract, methods, …).
* Targets *chunk_size* tokens, respecting sentence boundaries.
* Carries an *overlap* tail from the previous chunk so cross-chunk context
  is preserved during retrieval.
* Records the first page it came from and deterministic SHA-256 chunk IDs.

After filtering (done externally), call :func:`wire_neighbors` once to set
``prev_chunk_id`` / ``next_chunk_id`` on the surviving list.

Public API
----------
PageText                – Lightweight page container (text + page_number).
chunk_pages(...)        – Main entry point: pages → chunk dicts.
wire_neighbors(chunks)  – Wire doubly-linked IDs after external filtering.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from rag_system.utils.preprocessing import clean_text, token_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

#: Matches common scientific paper section headings, with or without numbering.
SECTION_RE = re.compile(
    r"^\s*"
    r"(?:\d+(?:\.\d+)*[\s.\-]*)?"          # optional leading "1.", "2.1", …
    r"(?P<name>"
    r"abstract|introduction|background|"
    r"methods?|materials?\s+and\s+methods?|methodology|"
    r"results?|"
    r"discussion|"
    r"conclusions?|summary"
    r")"
    r"\s*[:.]?\s*$",
    re.IGNORECASE,
)

#: Canonical aliases — map raw heading text → normalised section label.
SECTION_ALIASES: dict[str, str] = {
    "background":            "introduction",
    "method":                "methods",
    "materials and method":  "methods",
    "materials and methods": "methods",
    "methodology":           "methods",
    "result":                "results",
    "conclusions":           "conclusion",
    "summary":               "conclusion",
}

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

#: Split on sentence-ending punctuation followed by whitespace + capital letter,
#: or on paragraph breaks (one or more blank lines / newlines).
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")

# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------

#: Inclusive token-size bounds enforced in :func:`chunk_pages`.
_MIN_CHUNK_SIZE: int = 50
_MAX_CHUNK_SIZE: int = 2000

#: The overlap window must stay strictly below the chunk size.
_MAX_OVERLAP_RATIO: float = 0.5

_DEFAULT_SECTION: str = "unknown"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PageText:
    """Lightweight container for a single PDF page.

    Attributes
    ----------
    text:        Raw text extracted from the page (may be empty or noisy).
    page_number: 1-based page index within the source PDF.
    """

    text: str
    page_number: int

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"page_number must be ≥ 1, got {self.page_number}")
        self.text = self.text or ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_section(raw: str) -> str:
    """Return the normalised section name for *raw*, or *raw* itself."""
    key = raw.lower().strip()
    return SECTION_ALIASES.get(key, key)


def _chunk_id(source: str, ordinal: int, text: str) -> str:
    """Deterministic 20-character hex ID derived from source + ordinal + text prefix."""
    payload = f"{source}|{ordinal}|{text[:80]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _split_into_sentences(text: str) -> list[str]:
    """Split *text* on sentence boundaries and paragraph breaks."""
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Stage 1 – page → (sentence, section, page_number) units
# ---------------------------------------------------------------------------


def _pages_to_units(pages: list[PageText]) -> list[tuple[str, str, int]]:
    """Convert sorted pages to a flat list of ``(sentence, section, page_number)`` triples.

    Section headings are detected line-by-line; the current section label is
    carried forward until a new heading is encountered.
    """
    units: list[tuple[str, str, int]] = []
    section: str = _DEFAULT_SECTION

    for page in sorted(pages, key=lambda p: p.page_number):
        pending_lines: list[str] = []

        for line in page.text.splitlines():
            match = SECTION_RE.match(line)
            if match:
                # Flush accumulated lines under the current section.
                if pending_lines:
                    body = clean_text(" ".join(pending_lines))
                    for sentence in _split_into_sentences(body):
                        units.append((sentence, section, page.page_number))
                    pending_lines = []
                section = _canonical_section(match.group("name"))
            else:
                pending_lines.append(line)

        # Flush remaining lines for this page.
        if pending_lines:
            body = clean_text(" ".join(pending_lines))
            for sentence in _split_into_sentences(body):
                units.append((sentence, section, page.page_number))

    return units


# ---------------------------------------------------------------------------
# Stage 2 – units → chunks
# ---------------------------------------------------------------------------


def _units_to_chunks(
    units: list[tuple[str, str, int]],
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Pack sentence units into token-bounded, section-homogeneous chunks.

    Algorithm
    ---------
    1.  Walk forward from *start*, accumulating sentences into the current
        chunk until the token budget is reached or the section changes.
    2.  At section boundaries, start fresh (no overlap carried across sections).
    3.  Within a section, rewind *overlap* tokens from the end of the previous
        chunk to seed the next one.

    Each produced chunk dict contains:

    ``chunk_id``      – deterministic 20-char hex ID.
    ``text``          – cleaned, joined chunk text.
    ``section``       – normalised section label.
    ``source``        – PDF filename (stem or full name as supplied).
    ``page_number``   – page where the chunk starts.
    ``prev_chunk_id`` – set to ``None`` here; populated by :func:`wire_neighbors`.
    ``next_chunk_id`` – set to ``None`` here; populated by :func:`wire_neighbors`.
    ``ordinal``       – sequential index within this source (before filtering).
    ``token_count``   – number of tokens in *text* (cached for later filtering).
    """
    chunks: list[dict] = []
    n = len(units)
    start = 0

    while start < n:
        current_section = units[start][1]
        end = start
        size = 0

        # Accumulate sentences within the same section.
        while end < n and units[end][1] == current_section:
            cost = token_count(units[end][0])
            # Always include at least one sentence even if it alone exceeds the budget.
            if end > start and size + cost > chunk_size:
                break
            size += cost
            end += 1

        # Safety: never stall.
        if end == start:
            end += 1

        text = clean_text(" ".join(unit[0] for unit in units[start:end]))
        if text:
            ordinal = len(chunks)
            chunks.append(
                {
                    "chunk_id":      _chunk_id(source, ordinal, text),
                    "text":          text,
                    "section":       current_section,
                    "source":        source,
                    "page_number":   int(units[start][2]),
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                    "ordinal":       ordinal,
                    "token_count":   token_count(text),
                }
            )

        if end >= n:
            break

        # Section boundary → start fresh, no overlap.
        if units[end][1] != current_section:
            start = end
            continue

        # Within-section overlap: rewind until we've carried *overlap* tokens.
        rewind = end
        carried = 0
        while rewind > start and carried < overlap:
            rewind -= 1
            carried += token_count(units[rewind][0])

        start = rewind if rewind > start else end

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_pages(
    pages: list[PageText],
    source: str,
    chunk_size: int = 400,
    overlap: int = 64,
) -> list[dict]:
    """Convert a list of :class:`PageText` objects into sentence-level chunk dicts.

    Parameters
    ----------
    pages:      Pages extracted from a single PDF (order is by page_number).
    source:     PDF filename used as provenance in every chunk (e.g. ``"12345678.pdf"``).
    chunk_size: Target chunk size in tokens.  Must be in
                [``_MIN_CHUNK_SIZE``, ``_MAX_CHUNK_SIZE``].
    overlap:    Token overlap carried from the end of one chunk into the start of
                the next within the same section.  Must be ≥ 0, < *chunk_size*,
                and ≤ ``_MAX_OVERLAP_RATIO × chunk_size``.

    Returns
    -------
    List of chunk dicts.  ``prev_chunk_id`` and ``next_chunk_id`` are ``None``
    until :func:`wire_neighbors` is called (after any external filtering).

    Raises
    ------
    ValueError
        If *chunk_size* or *overlap* are outside their valid ranges.
    """
    # --- Input validation ---------------------------------------------------
    if not (_MIN_CHUNK_SIZE <= chunk_size <= _MAX_CHUNK_SIZE):
        raise ValueError(
            f"chunk_size must be in [{_MIN_CHUNK_SIZE}, {_MAX_CHUNK_SIZE}], got {chunk_size}"
        )
    if overlap < 0:
        raise ValueError(f"overlap must be ≥ 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
        )
    if overlap > _MAX_OVERLAP_RATIO * chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be ≤ {_MAX_OVERLAP_RATIO:.0%} of chunk_size ({chunk_size}); "
            f"got {overlap / chunk_size:.0%}"
        )
    if not source or not source.strip():
        raise ValueError("source must be a non-empty string")

    if not pages:
        logger.warning("chunk_pages: no pages supplied for source '%s'; returning []", source)
        return []

    # --- Pipeline -----------------------------------------------------------
    units = _pages_to_units(pages)
    if not units:
        logger.warning(
            "chunk_pages: no usable sentence units extracted from %d page(s) of '%s'",
            len(pages),
            source,
        )
        return []

    chunks = _units_to_chunks(units, source, chunk_size, overlap)

    logger.debug(
        "chunk_pages: source=%r pages=%d units=%d chunks=%d",
        source,
        len(pages),
        len(units),
        len(chunks),
    )
    return chunks


def wire_neighbors(chunks: list[dict]) -> None:
    """Set ``prev_chunk_id`` / ``next_chunk_id`` on a (possibly filtered) chunk list.

    Must be called **after** any quality filtering so that every stored
    neighbor reference is resolvable in the payload.  Also re-sequences the
    ``ordinal`` field to reflect the post-filter order.

    Parameters
    ----------
    chunks: In-place mutated list of chunk dicts.  No-op on an empty list.
    """
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk["ordinal"] = i
        chunk["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
        chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i + 1 < n else None
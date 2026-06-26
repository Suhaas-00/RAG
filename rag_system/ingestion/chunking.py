"""Sentence-, section-, and metadata-aware PDF chunking with page provenance.

Overview
--------
The chunker converts a list of :class:`PageText` objects (one per PDF page)
into a flat list of chunk dicts ready for embedding.  Two complementary
chunking strategies are produced and merged:

**Strategy A – Content chunks** (original, improved)
    Sentence-boundary-respecting, section-homogeneous windows that target
    *chunk_size* tokens with an *overlap* tail carried across consecutive
    windows within the same section.

**Strategy B – Metadata chunks**
    Dedicated short chunks that capture the abstract, title-page metadata
    (DOI, PMID, year, journal), disease list, gene list, and study-design
    keywords.  These are prefixed with structured field names so that
    keyword retrieval and exact-match filters work reliably even when the
    LLM embedding might miss a rare term.

Both strategy outputs are tagged with ``chunk_type`` (``"content"`` or
``"metadata"``) and share the same downstream schema.  After filtering (done
externally) call :func:`wire_neighbors` once to set
``prev_chunk_id`` / ``next_chunk_id`` on the surviving list.

Public API
----------
PageText                         – Lightweight page container (text + page_number).
chunk_pages(pages, source, ...)  – Main entry: pages → chunk dicts (both strategies).
wire_neighbors(chunks)           – Wire doubly-linked IDs after external filtering.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Optional

from rag_system.cleaner import clean_chunk_text, clean_pdf_pages, detect_section_heading
from rag_system.utils.preprocessing import clean_text, extract_metadata_from_text, token_count

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
    r"methods?|materials?\s+and\s+methods?|patients\s+and\s+methods|methodology|"
    r"study\s+design|experimental\s+procedures|"
    r"results?|"
    r"findings|discussion|"
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
    "patients and methods":  "methods",
    "methodology":           "methods",
    "experimental procedures": "methods",
    "study design":          "study design",
    "result":                "results",
    "findings":              "results",
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

_MIN_CHUNK_SIZE: int = 50
_MAX_CHUNK_SIZE: int = 2000

#: The overlap window must stay strictly below the chunk size.
_MAX_OVERLAP_RATIO: float = 0.5

_DEFAULT_SECTION: str = "unknown"

# Maximum token budget for a single metadata chunk field value.
_META_FIELD_MAX_TOKENS: int = 300


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
# Strategy A – helpers
# ---------------------------------------------------------------------------


def _pages_to_units(pages: list[PageText]) -> list[tuple[str, str, int]]:
    """Convert sorted pages to a flat list of ``(sentence, section, page_number)`` triples.

    Section headings are detected line-by-line; the current section label is
    carried forward until a new heading is encountered.
    """
    units: list[tuple[str, str, int]] = []
    section: str = _DEFAULT_SECTION

    cleaned_pages = clean_pdf_pages(pages)

    for page in sorted(cleaned_pages, key=lambda p: p.page):
        pending_lines: list[str] = []

        for line in page.text.splitlines():
            detected = detect_section_heading(line)
            match = SECTION_RE.match(line)
            if detected or match:
                # Flush accumulated lines under the current section.
                if pending_lines:
                    body = clean_chunk_text(" ".join(pending_lines))
                    for sentence in _split_into_sentences(body):
                        units.append((sentence, section, page.page))
                    pending_lines = []
                section = detected or _canonical_section(match.group("name"))
            else:
                pending_lines.append(line)

        # Flush remaining lines for this page.
        if pending_lines:
            body = clean_chunk_text(" ".join(pending_lines))
            for sentence in _split_into_sentences(body):
                units.append((sentence, section, page.page))

    return units


def _units_to_chunks(
    units: list[tuple[str, str, int]],
    source: str,
    chunk_size: int,
    overlap: int,
    ordinal_start: int = 0,
) -> list[dict]:
    """Pack sentence units into token-bounded, section-homogeneous chunks.

    Algorithm
    ---------
    1. Walk forward from *start*, accumulating sentences into the current
       chunk until the token budget is reached or the section changes.
    2. At section boundaries, start fresh (no overlap carried across sections).
    3. Within a section, rewind *overlap* tokens from the end of the previous
       chunk to seed the next one.

    Each produced chunk dict contains:

    ``chunk_id``      – deterministic 20-char hex ID.
    ``text``          – cleaned, joined chunk text.
    ``section``       – normalised section label.
    ``chunk_type``    – always ``"content"`` for this strategy.
    ``source``        – PDF filename.
    ``page_number``   – page where the chunk starts.
    ``prev_chunk_id`` – ``None`` until :func:`wire_neighbors` is called.
    ``next_chunk_id`` – ``None`` until :func:`wire_neighbors` is called.
    ``ordinal``       – sequential index within this source (before filtering).
    ``token_count``   – number of tokens in *text*.
    ``metadata``      – empty dict placeholder (populated during ingestion).
    """
    chunks: list[dict] = []
    n = len(units)
    start = 0
    ordinal = ordinal_start

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
            chunks.append(
                {
                    "chunk_id":      _chunk_id(source, ordinal, text),
                    "text":          text,
                    "section":       current_section,
                    "chunk_type":    "content",
                    "source":        source,
                    "page_number":   int(units[start][2]),
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                    "ordinal":       ordinal,
                    "token_count":   token_count(text),
                    "metadata":      {},
                }
            )
            ordinal += 1

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
# Strategy B – metadata chunk builders
# ---------------------------------------------------------------------------


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return *text* truncated so that token_count ≤ max_tokens."""
    if token_count(text) <= max_tokens:
        return text
    words = text.split()
    result: list[str] = []
    running = 0
    for word in words:
        cost = token_count(word)
        if running + cost > max_tokens:
            break
        result.append(word)
        running += cost
    return " ".join(result)


def _build_metadata_chunks(
    pages: list[PageText],
    source: str,
    doc_metadata: dict,
    ordinal_start: int = 0,
) -> list[dict]:
    """Build dedicated metadata chunks for a document.

    One chunk is created per metadata field that has content.  The text is
    formatted as ``"<FIELD>: <value>"`` so that keyword and embedding
    retrieval both work on the structured representation.

    Additionally, an **abstract chunk** is extracted from the first two pages
    if any page contains an ``abstract`` section heading.

    Parameters
    ----------
    pages:          All pages of the document (sorted by page_number externally).
    source:         PDF filename used as provenance.
    doc_metadata:   Pre-extracted metadata dict from
                    :func:`~rag_system.utils.preprocessing.extract_metadata_from_text`.
    ordinal_start:  Starting ordinal so metadata chunks get unique IDs even
                    when content chunks are produced first.

    Returns
    -------
    List of metadata chunk dicts.  Never returns ``None`` but may be empty.
    """
    chunks: list[dict] = []
    ordinal = ordinal_start

    def _meta_chunk(field: str, value: str, section: str = "metadata") -> dict:
        text = _truncate_to_tokens(f"{field}: {value}", _META_FIELD_MAX_TOKENS)
        return {
            "chunk_id":      _chunk_id(source, ordinal, text),
            "text":          text,
            "section":       section,
            "chunk_type":    "metadata",
            "source":        source,
            "page_number":   1,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "ordinal":       ordinal,
            "token_count":   token_count(text),
            "metadata":      {k: v for k, v in doc_metadata.items()},
        }

    # ── Bibliographic fields ──────────────────────────────────────────────
    if doc_metadata.get("doi"):
        chunks.append(_meta_chunk("DOI", doc_metadata["doi"]))
        ordinal += 1

    if doc_metadata.get("pmid"):
        chunks.append(_meta_chunk("PubMed ID", doc_metadata["pmid"]))
        ordinal += 1

    if doc_metadata.get("year"):
        chunks.append(_meta_chunk("Publication year", doc_metadata["year"]))
        ordinal += 1

    if doc_metadata.get("journal"):
        chunks.append(_meta_chunk("Journal", doc_metadata["journal"]))
        ordinal += 1

    # ── Disease / gene / study-design fields ─────────────────────────────
    diseases = doc_metadata.get("diseases", [])
    if diseases:
        value = "; ".join(diseases)
        chunks.append(_meta_chunk("Diseases mentioned", value, section="metadata"))
        ordinal += 1

    genes = doc_metadata.get("genes", [])
    if genes:
        value = "; ".join(genes)
        chunks.append(_meta_chunk("Genes and biomarkers mentioned", value, section="metadata"))
        ordinal += 1

    study_designs = doc_metadata.get("study_designs", [])
    if study_designs:
        value = "; ".join(study_designs)
        chunks.append(_meta_chunk("Study design", value, section="metadata"))
        ordinal += 1

    # ── Abstract chunk ────────────────────────────────────────────────────
    # Collect lines from first 3 pages that fall under the abstract section.
    abstract_lines: list[str] = []
    in_abstract = False
    for page in sorted(pages, key=lambda p: p.page_number)[:3]:
        for line in page.text.splitlines():
            match = SECTION_RE.match(line)
            if match:
                sec = _canonical_section(match.group("name"))
                in_abstract = sec == "abstract"
            elif in_abstract:
                abstract_lines.append(line.strip())

    if not abstract_lines:
        # Fallback: use the first ~400 tokens from page 1 as a pseudo-abstract.
        first_page_text = pages[0].text if pages else ""
        abstract_lines = [first_page_text[:2000]]

    abstract_text = clean_text(" ".join(abstract_lines))
    abstract_text = _truncate_to_tokens(abstract_text, 400)
    if abstract_text:
        text = f"Abstract: {abstract_text}"
        chunks.append(
            {
                "chunk_id":      _chunk_id(source, ordinal, text),
                "text":          text,
                "section":       "abstract",
                "chunk_type":    "metadata",
                "source":        source,
                "page_number":   1,
                "prev_chunk_id": None,
                "next_chunk_id": None,
                "ordinal":       ordinal,
                "token_count":   token_count(text),
                "metadata":      {k: v for k, v in doc_metadata.items()},
            }
        )
        ordinal += 1

    logger.debug(
        "_build_metadata_chunks: source=%r produced %d metadata chunks",
        source,
        len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_pages(
    pages: list[PageText],
    source: str,
    chunk_size: int = 400,
    overlap: int = 64,
    doc_metadata: Optional[dict] = None,
) -> list[dict]:
    """Convert a list of :class:`PageText` objects into chunk dicts.

    Produces **both** content chunks (Strategy A) and metadata chunks
    (Strategy B) and returns them in a single merged list.  Callers should
    filter and call :func:`wire_neighbors` afterwards.

    Parameters
    ----------
    pages:
        Pages extracted from a single PDF (order is by page_number).
    source:
        PDF filename used as provenance in every chunk.
    chunk_size:
        Target chunk size in tokens.  Must be in
        [``_MIN_CHUNK_SIZE``, ``_MAX_CHUNK_SIZE``].
    overlap:
        Token overlap carried from the end of one chunk into the start of
        the next within the same section.  Must be ≥ 0, < *chunk_size*,
        and ≤ ``_MAX_OVERLAP_RATIO × chunk_size``.
    doc_metadata:
        Pre-extracted document metadata dict.  When ``None``, metadata is
        derived here from the concatenation of all page texts.

    Returns
    -------
    Merged list of content chunk dicts followed by metadata chunk dicts.
    ``prev_chunk_id`` and ``next_chunk_id`` are ``None`` until
    :func:`wire_neighbors` is called.

    Raises
    ------
    ValueError
        If *chunk_size* or *overlap* are outside their valid ranges, or
        if *source* is empty.
    """
    # --- Input validation ---------------------------------------------------
    if not (_MIN_CHUNK_SIZE <= chunk_size <= _MAX_CHUNK_SIZE):
        raise ValueError(
            f"chunk_size must be in [{_MIN_CHUNK_SIZE}, {_MAX_CHUNK_SIZE}], got {chunk_size}"
        )
    if overlap < 0:
        raise ValueError(f"overlap must be ≥ 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
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

    # --- Extract document-level metadata if not supplied --------------------
    if doc_metadata is None:
        full_text = "\n".join(p.text for p in sorted(pages, key=lambda p: p.page_number))
        doc_metadata = extract_metadata_from_text(full_text)
        logger.debug(
            "chunk_pages: auto-extracted metadata for '%s': diseases=%s genes=%s",
            source,
            doc_metadata.get("diseases"),
            doc_metadata.get("genes"),
        )

    # --- Strategy A: content chunks -----------------------------------------
    units = _pages_to_units(pages)
    if not units:
        logger.warning(
            "chunk_pages: no usable sentence units from %d page(s) of '%s'",
            len(pages),
            source,
        )
        content_chunks: list[dict] = []
    else:
        content_chunks = _units_to_chunks(units, source, chunk_size, overlap, ordinal_start=0)
        # Attach document metadata to every content chunk for filtering later.
        for chunk in content_chunks:
            chunk["metadata"] = doc_metadata

    # --- Strategy B: metadata chunks ----------------------------------------
    meta_chunks = _build_metadata_chunks(
        pages,
        source,
        doc_metadata,
        ordinal_start=len(content_chunks),
    )

    all_chunks = content_chunks + meta_chunks

    logger.debug(
        "chunk_pages: source=%r pages=%d units=%d content=%d metadata=%d total=%d",
        source,
        len(pages),
        len(units),
        len(content_chunks),
        len(meta_chunks),
        len(all_chunks),
    )
    return all_chunks


def wire_neighbors(chunks: list[dict]) -> None:
    """Set ``prev_chunk_id`` / ``next_chunk_id`` on a (possibly filtered) chunk list.

    Only **content** chunks are wired together; metadata chunks are
    self-contained and do not benefit from neighbour navigation.

    Must be called **after** any quality filtering so that every stored
    neighbour reference is resolvable in the payload.  Also re-sequences the
    ``ordinal`` field to reflect the post-filter order.

    Parameters
    ----------
    chunks: In-place mutated list of chunk dicts.  No-op on an empty list.
    """
    content = [c for c in chunks if c.get("chunk_type") != "metadata"]
    n = len(content)
    for i, chunk in enumerate(content):
        chunk["prev_chunk_id"] = content[i - 1]["chunk_id"] if i > 0 else None
        chunk["next_chunk_id"] = content[i + 1]["chunk_id"] if i + 1 < n else None

    # Re-sequence ordinals across all chunks (content + metadata).
    for i, chunk in enumerate(chunks):
        chunk["ordinal"] = i

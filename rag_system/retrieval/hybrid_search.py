"""Source-constrained FAISS retrieval with semantic and keyword scoring.

Public API
----------
hybrid_search(query, embedder, index, payload, ...)
    Full hybrid retrieval pipeline; returns ranked candidate dicts.

is_noisy(text)
    Predicate that identifies low-quality OCR / structural text.

keyword_score(query, text)
    Normalised Jaccard-style keyword overlap score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from rag_system.utils.preprocessing import normalize_for_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text-quality predicates
# ---------------------------------------------------------------------------

# Structural markers that indicate the chunk is a table, figure caption,
# reference list, or download banner — not useful for free-text QA.
_NOISE_MARKERS: frozenset[str] = frozenset(
    {"table", "figure", "downloaded from", "references", "supplementary"}
)

# Chunks where more than 30 % of characters are digits are likely OCR numeric
# data (e.g. statistical tables) rather than prose.
_MAX_DIGIT_RATIO: float = 0.30


def is_noisy(text: Optional[str]) -> bool:
    """Return ``True`` for chunks that are structural noise rather than prose.

    Criteria
    --------
    * Contains a noise marker phrase (table, figure, …).
    * Digit-to-character ratio exceeds :data:`_MAX_DIGIT_RATIO`.

    Parameters
    ----------
    text: Raw chunk text from the metadata payload.
    """
    lowered = (text or "").lower()
    if any(marker in lowered for marker in _NOISE_MARKERS):
        return True
    length = max(len(lowered), 1)
    return sum(ch.isdigit() for ch in lowered) / length > _MAX_DIGIT_RATIO


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------


def keyword_score(query: str, text: str) -> float:
    """Normalised keyword overlap between *query* and *text*.

    Computes  |Q ∩ T| / (|Q| + 1)  on normalised token sets.

    Returns a float in ``[0.0, 1.0)``.  Adding 1 to the denominator avoids
    division-by-zero and penalises very short queries less harshly.

    Parameters
    ----------
    query: The user query (will be normalised internally).
    text:  The chunk text (will be normalised internally).
    """
    q_words = set(normalize_for_embedding(query).split())
    t_words = set(normalize_for_embedding(text).split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / (len(q_words) + 1)


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def _source_positions(
    records: list[dict],
    source_filter: Optional[str],
    section_filter: Optional[str] = None,
) -> list[int]:
    """Return a list of record indices that match the optional constraints.

    Filtering strategy
    ------------------
    1. Remove noisy records unconditionally.
    2. If *section_filter* is set, keep only records whose ``section`` field
       matches (case-insensitive).
    3. If *source_filter* is set, attempt an exact filename match first; fall
       back to a stem match for legacy metadata that omits the ``.pdf`` suffix.

    Parameters
    ----------
    records:        Full list of metadata dicts from the payload.
    source_filter:  PDF filename (``"12345678.pdf"``) or ``None``.
    section_filter: Section label (``"abstract"``) or ``None``.
    """
    # Step 1 – drop structural noise.
    clean_indices: list[int] = [
        i for i, r in enumerate(records) if not is_noisy(r.get("text", ""))
    ]

    # Step 2 – optional section constraint.
    if section_filter:
        wanted_section = section_filter.casefold()
        clean_indices = [
            i for i in clean_indices
            if str(records[i].get("section", "")).casefold() == wanted_section
        ]

    # Step 3 – optional source constraint.
    if not source_filter:
        return clean_indices

    wanted = Path(source_filter).name.casefold()

    # Exact filename match (preferred).
    exact = [
        i for i in clean_indices
        if Path(str(records[i].get("source", ""))).name.casefold() == wanted
    ]
    if exact:
        return exact

    # Stem-only match for legacy metadata without the ``.pdf`` extension.
    wanted_stem = Path(wanted).stem
    return [
        i for i in clean_indices
        if Path(str(records[i].get("source", ""))).stem.casefold() == wanted_stem
    ]


# ---------------------------------------------------------------------------
# FAISS subset search
# ---------------------------------------------------------------------------


def _faiss_subset_search(
    query_vector: np.ndarray,
    index: faiss.Index,
    positions: list[int],
    k: int,
) -> list[tuple[int, float]]:
    """Search a temporary in-memory IndexFlatIP built from *positions* only.

    Building a temporary index at query time allows arbitrary source/section
    filtering without re-training or rebuilding the main index.

    Parameters
    ----------
    query_vector: Shape ``(1, d)`` float32 array.
    index:        The full FAISS index (used only for ``reconstruct``).
    positions:    Allowed row indices into the full index.
    k:            Number of neighbours to return.

    Returns
    -------
    List of ``(position_in_full_index, inner_product_score)`` tuples, ordered
    by descending score.
    """
    if not positions:
        return []

    # Reconstruct and L2-normalise the subset vectors.
    vectors = np.vstack(
        [index.reconstruct(int(pos)) for pos in positions]
    ).astype("float32")
    faiss.normalize_L2(vectors)

    subset = faiss.IndexFlatIP(index.d)
    subset.add(vectors)

    actual_k = min(k, len(positions))
    scores, local_ids = subset.search(query_vector, actual_k)

    return [
        (positions[int(lid)], float(score))
        for lid, score in zip(local_ids[0], scores[0])
        if lid >= 0
    ]


# ---------------------------------------------------------------------------
# Public hybrid search
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    embedder,
    index: faiss.Index,
    payload: dict,
    candidate_k: int = 10,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    source_filter: Optional[str] = None,
    section_filter: Optional[str] = None,
) -> list[dict]:
    """Return the top *candidate_k* hybrid-scored chunks for *query*.

    Pipeline
    --------
    1. Encode *query* to a dense vector with *embedder*.
    2. Build an allowed-position list using *source_filter* and *section_filter*.
    3. Run FAISS inner-product search on the filtered subset.
    4. If a source filter was requested but produced zero matches, fall back to
       the whole clean corpus (retaining any section constraint).
    5. Compute per-candidate hybrid scores and return the enriched dicts.

    Parameters
    ----------
    query:          Cleaned, lowercase query string.
    embedder:       Object with an ``encode(texts) → np.ndarray`` method.
    index:          Loaded FAISS index (``faiss.Index`` subclass).
    payload:        Metadata dict from ``metadata.pkl``; must contain
                    ``"records"`` (list of chunk dicts).
    candidate_k:    Number of nearest neighbours to retrieve before reranking.
    semantic_weight: Weight for the semantic (inner-product) score component.
    keyword_weight:  Weight for the keyword overlap score component.
    source_filter:  Optional PDF filename to restrict retrieval.
    section_filter: Optional section name to restrict retrieval.

    Returns
    -------
    List of chunk dicts (copies), each enriched with:

    ``semantic_score_raw``  – raw FAISS inner-product score.
    ``semantic_score``      – same as raw (alias for reranker compatibility).
    ``keyword_score_raw``   – keyword overlap score.
    ``keyword_score``       – same as raw.
    ``hybrid_score``        – weighted combination.
    """
    records: list[dict] = payload.get("records", [])
    if not records:
        logger.warning("hybrid_search: payload contains no records.")
        return []

    # Encode query to a unit-normalised float32 vector.
    query_vector: np.ndarray = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    positions = _source_positions(records, source_filter, section_filter)
    logger.debug(
        "hybrid_search: source_filter=%r section_filter=%r → %d positions",
        source_filter,
        section_filter,
        len(positions),
    )

    matches = _faiss_subset_search(query_vector, index, positions, candidate_k)

    # Required fallback: when a source filter yields nothing, search the full
    # clean corpus (honouring any section constraint that remains useful).
    if source_filter and not matches:
        logger.info(
            "hybrid_search: source filter '%s' matched 0 positions; falling back to full corpus.",
            source_filter,
        )
        positions = _source_positions(records, None, section_filter)
        matches = _faiss_subset_search(query_vector, index, positions, candidate_k)

    results: list[dict] = []
    for position, semantic in matches:
        record = dict(records[position])
        kw = keyword_score(query, record.get("text", ""))
        record.update(
            {
                "semantic_score_raw": semantic,
                "semantic_score": semantic,
                "keyword_score_raw": kw,
                "keyword_score": kw,
                "hybrid_score": semantic_weight * semantic + keyword_weight * kw,
            }
        )
        results.append(record)

    logger.debug(
        "hybrid_search: returning %d candidates (top hybrid=%.4f)",
        len(results),
        max((r["hybrid_score"] for r in results), default=float("nan")),
    )

    return results
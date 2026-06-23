"""Section-aware reranking for clean FAISS candidates.

The reranker applies a weighted combination of semantic and keyword scores,
then adds a section-position bias that reflects how informative each section
typically is for question-answering over scientific papers.

Scoring formula
---------------
    final_score = 0.7 × semantic_score + 0.3 × keyword_score + section_boost(section)

The 0.7 / 0.3 split mirrors the hybrid-search weights so that reranking
refines rather than contradicts the retrieval order.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag_system.retrieval.hybrid_search import is_noisy, keyword_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section boost table
# ---------------------------------------------------------------------------

# Positive bias for sections that tend to directly answer factual questions;
# negative bias for sections that are typically verbose or tangential.
_SECTION_BOOSTS: dict[str, float] = {
    "abstract":     0.25,
    "introduction": 0.20,
    "results":      0.15,
    "methods":      0.10,
    "conclusion":   0.05,
    "discussion":  -0.10,
    "table":       -0.15,
    "figure":      -0.15,
    "references":  -0.20,
}

_DEFAULT_BOOST: float = 0.0

# Weights for the hybrid reranking formula.
_SEMANTIC_WEIGHT: float = 0.7
_KEYWORD_WEIGHT: float = 0.3


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def section_boost(section: Optional[str]) -> float:
    """Return a score adjustment based on the paper section.

    Parameters
    ----------
    section:
        The section label stored in chunk metadata (case-insensitive).
        ``None`` or an unrecognised section returns 0.0.

    Returns
    -------
    A float in the range ``[-0.20, +0.25]``.
    """
    return _SECTION_BOOSTS.get((section or "").lower().strip(), _DEFAULT_BOOST)


def rerank(
    query: str,
    candidates: list[dict],
    *,
    top_k: int = 3,
    semantic_weight: float = _SEMANTIC_WEIGHT,
    keyword_weight: float = _KEYWORD_WEIGHT,
) -> list[dict]:
    """Score, filter, and rank *candidates* returning at most *top_k* results.

    Steps
    -----
    1. Drop noisy chunks (tables, figures, OCR debris).
    2. Re-compute keyword score against the cleaned query.
    3. Apply the weighted hybrid formula plus a section bias.
    4. Sort descending by ``final_score`` and truncate to *top_k*.

    Parameters
    ----------
    query:
        The cleaned query string (same normalisation as used during retrieval).
    candidates:
        List of chunk dicts produced by :func:`hybrid_search`.  Each dict must
        contain at least ``"text"``, ``"semantic_score"``, and ``"section"``.
    top_k:
        Maximum number of results to return.
    semantic_weight:
        Weight applied to the semantic score component.
    keyword_weight:
        Weight applied to the keyword score component.

    Returns
    -------
    A list of dicts (copies of candidate dicts with added scoring keys),
    sorted descending by ``final_score``.  Each result carries:

    ``keyword_score``   – normalised keyword overlap score for *query*.
    ``final_score``     – weighted combination used for ranking.
    ``rerank_score``    – alias for ``final_score`` (backward-compatible).
    ``section_boost``   – the bias added for this chunk's section.
    """
    if not candidates:
        return []

    if top_k < 1:
        raise ValueError(f"top_k must be ≥ 1, got {top_k}")
    if not (0.0 <= semantic_weight <= 1.0):
        raise ValueError(f"semantic_weight must be in [0, 1], got {semantic_weight}")
    if not (0.0 <= keyword_weight <= 1.0):
        raise ValueError(f"keyword_weight must be in [0, 1], got {keyword_weight}")

    ranked: list[dict] = []
    dropped_noisy = 0

    for item in candidates:
        text = item.get("text", "")
        if is_noisy(text):
            dropped_noisy += 1
            continue

        kw_score = keyword_score(query, text)
        sec_boost = section_boost(item.get("section"))
        score = semantic_weight * item.get("semantic_score", 0.0) + keyword_weight * kw_score + sec_boost

        ranked.append(
            {
                **item,
                "keyword_score": kw_score,
                "keyword_score_raw": item.get("keyword_score_raw", kw_score),
                "section_boost": sec_boost,
                "final_score": score,
                "rerank_score": score,           # Backward-compatible alias.
            }
        )

    if dropped_noisy:
        logger.debug("rerank: dropped %d noisy candidate(s)", dropped_noisy)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    result = ranked[:top_k]

    logger.debug(
        "rerank: %d candidates → %d ranked (top score=%.4f)",
        len(candidates),
        len(result),
        result[0]["final_score"] if result else float("nan"),
    )

    return result
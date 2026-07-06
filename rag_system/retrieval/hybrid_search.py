"""Source-constrained hybrid retrieval with full enhancement stack.

This module extends the original ``hybrid_search`` implementation with every
technique that was previously missing:

* Reciprocal Rank Fusion (RRF)               -> ``reciprocal_rank_fusion``
* Multi-query retrieval                       -> ``multi_query_retrieve``
* Clinical synonym expansion                  -> ``expand_query_synonyms``
* Query classification & routing              -> ``classify_query``, ``route_query``
* Cross-encoder reranking                     -> ``CrossEncoderReranker``
* Confidence-based fallback thresholds        -> ``retrieve_with_fallback``
* HyDE (Hypothetical Document Embeddings)     -> ``hyde_retrieve``

Document-scope integration (``document_scope.py``)
---------------------------------------------------
``document_scope`` is **not** imported or required by default. The module
runs fully standalone using its own inline filtering logic inside
``_source_positions``.

To opt-in to the richer ``document_scope`` filtering (disease, gene, year,
nested-metadata, multi-alias support) at any point in the future:

    # At the top of whatever entrypoint uses hybrid_search:
    from rag_system.retrieval.document_scope import filter_record_indices
    import rag_system.retrieval.hybrid_search as hs
    hs.enable_document_scope(filter_record_indices)

That single call swaps the inline filter for the full ``CompiledFilters``
implementation without touching any call site.

Public API
----------
hybrid_search(query, embedder, index, payload, ...)
    Original hybrid retrieval pipeline (semantic + keyword), sorted by
    fused score.

is_noisy(text)
    Predicate that identifies low-quality OCR / structural text.

keyword_score(query, text)
    Normalised keyword overlap score.

reciprocal_rank_fusion(ranked_lists, k=60)
    Fuse multiple ranked candidate lists by reciprocal rank.

expand_query_synonyms(query, synonym_map=None)
    Expand a query string with clinical-trial synonym terms.

classify_query(query) / route_query(query)
    Classify a question into a retrieval-strategy bucket and resolve it to
    concrete routing parameters.

CrossEncoderReranker
    Thin wrapper around ``sentence_transformers.CrossEncoder``.

hyde_retrieve(query, embedder, index, payload, llm, ...)
    Hypothetical Document Embeddings retrieval.

multi_query_retrieve(query, embedder, index, payload, llm, ...)
    Generate query reformulations via a pluggable LLM, retrieve for each,
    fuse with RRF.

retrieve_with_fallback(query, embedder, index, payload, ...)
    Full production pipeline: classify -> route -> retrieve -> rerank ->
    confidence-gate -> filter-relaxation fallback ladder.

enable_document_scope(filter_fn)
    Opt-in function that replaces the inline filter with an external
    ``filter_record_indices`` implementation (e.g. from document_scope.py).

disable_document_scope()
    Revert to the built-in inline filter.

LLMClient (Protocol)
    Interface multi-query / HyDE depend on.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Protocol

import faiss
import numpy as np

from rag_system.retrieval.retrieval_config import document_scope_enabled as config_document_scope_enabled
from rag_system.utils.preprocessing import normalize_for_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional document_scope hook
# ---------------------------------------------------------------------------
# Default: None. Set via enable_document_scope() to activate the full
# CompiledFilters pipeline from document_scope.py without changing any
# call sites in this module. document_scope.py is NEVER imported here.

_external_filter_fn: Optional[Callable[..., list[int]]] = None


def enable_document_scope(filter_fn: Callable[..., list[int]]) -> None:
    """Swap the inline position filter for an external implementation.

    Parameters
    ----------
    filter_fn:
        A callable with the same signature as
        ``document_scope.filter_record_indices``, i.e.::

            filter_fn(
                records,
                filters,                     # dict[str, Any]
                *,
                enable_document_filtering,   # bool
                allow_global_search,         # bool
                require_document_scope,      # bool
                drop_noisy,                  # bool
                noisy_predicate,             # Callable[[str], bool] | None
            ) -> list[int]

    Usage example (in your application entrypoint)::

        from rag_system.retrieval.document_scope import filter_record_indices
        from rag_system.retrieval import hybrid_search as hs_module
        hs_module.enable_document_scope(filter_record_indices)

    This is a process-level switch -- call once at startup; all subsequent
    retrieval calls in the same process will use the supplied function.
    """
    global _external_filter_fn
    _external_filter_fn = filter_fn
    logger.info(
        "hybrid_search: document_scope filtering enabled via %s",
        getattr(filter_fn, "__qualname__", repr(filter_fn)),
    )


def disable_document_scope() -> None:
    """Revert to the built-in inline filtering (useful in tests)."""
    global _external_filter_fn
    _external_filter_fn = None
    logger.info("hybrid_search: document_scope filtering disabled; using inline filter.")


def document_scope_enabled() -> bool:
    """Return ``True`` if an external document_scope filter is active."""
    return _external_filter_fn is not None


# ---------------------------------------------------------------------------
# Text-quality predicates (unchanged from original)
# ---------------------------------------------------------------------------

_NOISE_MARKERS: frozenset[str] = frozenset(
    {"table", "figure", "downloaded from", "references", "supplementary"}
)
_MAX_DIGIT_RATIO: float = 0.30


def is_noisy(text: Optional[str]) -> bool:
    """Return ``True`` for chunks that are structural noise rather than prose.

    Criteria
    --------
    * Contains a noise marker phrase (table, figure, ...).
    * Digit-to-character ratio exceeds :data:`_MAX_DIGIT_RATIO`.
    """
    lowered = (text or "").lower()
    if any(marker in lowered for marker in _NOISE_MARKERS):
        return True
    length = max(len(lowered), 1)
    return sum(ch.isdigit() for ch in lowered) / length > _MAX_DIGIT_RATIO


# ---------------------------------------------------------------------------
# Keyword scoring (unchanged from original)
# ---------------------------------------------------------------------------


def keyword_score(query: str, text: str) -> float:
    """Normalised keyword overlap between *query* and *text*.

    Computes  |Q ∩ T| / (|Q| + 1)  on normalised token sets.
    Returns a float in ``[0.0, 1.0)``.
    """
    q_words = set(normalize_for_embedding(query).split())
    t_words = set(normalize_for_embedding(text).split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / (len(q_words) + 1)


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def _inline_filter(
    records: list[dict],
    source_filter: Optional[str],
    section_filter: Optional[str],
    paper_id_filter: Optional[str],
    document_id_filter: Optional[str],
) -> list[int]:
    """Built-in position filter -- no external dependencies required.

    Replicates the original ``_source_positions`` logic extended with
    ``paper_id_filter`` and ``document_id_filter`` support.

    Filtering order
    ---------------
    1. Drop noisy records unconditionally (``is_noisy``).
    2. section_filter  -- case-insensitive exact match on ``record["section"]``.
    3. paper_id_filter -- checked against ``paper_id``, ``document_id``,
       ``doc_id``, ``source_id`` fields (any match suffices).
    4. document_id_filter -- same candidate keys as paper_id_filter.
    5. source_filter -- exact filename match first; falls back to stem match
       for legacy records that omit the ``.pdf`` suffix.

    If both a semantic-id filter (paper_id / document_id) and a source
    filter are provided, a record must satisfy ALL of them (AND logic),
    which is the conservative safe default for document-scoped retrieval.
    """
    _DOCUMENT_ID_KEYS = ("paper_id", "document_id", "doc_id", "source_id")
    _SOURCE_KEYS = ("source", "filename", "file_name")

    def _get(record: dict, key: str) -> Optional[str]:
        v = record.get(key)
        if v is None and isinstance(record.get("metadata"), dict):
            v = record["metadata"].get(key)
        return str(v).casefold() if v is not None else None

    def _id_matches(record: dict, wanted: str) -> bool:
        wf = wanted.casefold()
        return any(_get(record, k) == wf for k in _DOCUMENT_ID_KEYS)

    def _source_matches(record: dict, wanted: str) -> bool:
        wanted_path = Path(wanted)
        wanted_name = wanted_path.name.casefold()
        wanted_stem = wanted_path.stem.casefold()
        for key in _SOURCE_KEYS:
            actual = record.get(key)
            if actual is None:
                continue
            actual_path = Path(str(actual))
            if actual_path.name.casefold() == wanted_name:
                return True
            if actual_path.stem.casefold() == wanted_stem:
                return True
        return False

    result: list[int] = []
    for idx, record in enumerate(records):
        # Step 1 -- noise gate
        if is_noisy(record.get("text", "")):
            continue
        # Step 2 -- section gate
        if section_filter:
            if str(record.get("section", "")).casefold() != section_filter.casefold():
                continue
        # Step 3 -- paper_id gate
        if paper_id_filter and not _id_matches(record, paper_id_filter):
            continue
        # Step 4 -- document_id gate
        if document_id_filter and not _id_matches(record, document_id_filter):
            continue
        # Step 5 -- source gate
        if source_filter and not _source_matches(record, source_filter):
            continue
        result.append(idx)
    return result


def _source_positions(
    records: list[dict],
    source_filter: Optional[str],
    section_filter: Optional[str] = None,
    paper_id_filter: Optional[str] = None,
    document_id_filter: Optional[str] = None,
    allow_global_search: bool = True,
) -> list[int]:
    """Return record indices that survive all active scope filters.

    Delegates to :func:`_inline_filter` by default. When
    :func:`enable_document_scope` has been called, delegates to the
    registered external ``filter_record_indices`` instead, giving you the
    full ``CompiledFilters`` pipeline (disease/gene/year filters, nested
    metadata, multi-alias support) without any change to call sites.

    Parameters
    ----------
    records:            Full payload record list.
    source_filter:      PDF filename (``"12345678.pdf"``), or ``None``.
    section_filter:     Section label (``"abstract"``), or ``None``.
    paper_id_filter:    Paper / document ID, or ``None``.
    document_id_filter: Alternative document ID field, or ``None``.
    allow_global_search: When ``True`` and no filter resolves any position,
        the fallback (corpus-wide search) is permitted downstream. Passed
        through to the external filter when document_scope is enabled.
    """
    enable_scope = config_document_scope_enabled()

    # --- Path A: external document_scope.filter_record_indices -------------
    if _external_filter_fn is not None:
        return _external_filter_fn(
            records,
            {
                "source": source_filter,
                "section": section_filter,
                "paper_id": paper_id_filter,
                "document_id": document_id_filter,
            },
            enable_document_filtering=enable_scope,
            allow_global_search=allow_global_search,
            require_document_scope=not allow_global_search,
            drop_noisy=True,
            noisy_predicate=is_noisy,
        )

    # --- Path B: built-in inline filter (default, no external dependency) --
    positions = _inline_filter(
        records,
        source_filter if enable_scope else None,
        section_filter,
        paper_id_filter if enable_scope else None,
        document_id_filter if enable_scope else None,
    )

    # If any id/source filter was active but matched nothing, and global
    # search is not permitted, log and return empty rather than silently
    # returning the whole corpus.
    any_filter_active = any(
        f is not None for f in (source_filter, paper_id_filter, document_id_filter)
    )
    if any_filter_active and not positions and not allow_global_search:
        logger.debug(
            "_source_positions: filters produced 0 matches "
            "(source=%r paper_id=%r document_id=%r allow_global=%r)",
            source_filter, paper_id_filter, document_id_filter, allow_global_search,
        )

    return positions


# ---------------------------------------------------------------------------
# FAISS subset search (unchanged from original)
# ---------------------------------------------------------------------------


def _faiss_subset_search(
    query_vector: np.ndarray,
    index: faiss.Index,
    positions: list[int],
    k: int,
) -> list[tuple[int, float]]:
    """Search a temporary in-memory IndexFlatIP built from *positions* only."""
    if not positions:
        return []

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
# Public hybrid search (sorted by fused hybrid_score)
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
    paper_id_filter: Optional[str] = None,
    document_id_filter: Optional[str] = None,
    allow_global_search: bool | None = None,
) -> list[dict]:
    """Return the top *candidate_k* hybrid-scored chunks for *query*.

    Sorting fix
    -----------
    The original implementation returned candidates in raw FAISS semantic-
    score order even though it computed a separate ``hybrid_score``. This
    version sorts by ``hybrid_score`` descending before returning.

    Filtering
    ---------
    All four filter parameters (``source_filter``, ``section_filter``,
    ``paper_id_filter``, ``document_id_filter``) are optional -- pass only
    what you have. Document-scope filtering uses the built-in inline logic
    unless :func:`enable_document_scope` has been called first.

    Global-search auto-detection
    -----------------------------
    ``allow_global_search`` defaults to ``None`` (auto):

    * If the query is *unscoped* (no ``source_filter`` / ``paper_id_filter``
      / ``document_id_filter`` supplied at all), global corpus search is
      allowed automatically -- a user with no specific paper in mind still
      gets an answer instead of an empty result from the document-scope
      guard (relevant when :func:`enable_document_scope` is active).
    * If the query is scoped, auto mode stays strict (``False``): a scoped
      filter that matches nothing does not silently expand to the whole
      corpus.
    * Passing an explicit ``True``/``False`` always overrides auto-detection.

    Fallback
    --------
    When ``source_filter`` is set but produces zero matching positions,
    the function falls back to the full clean corpus (honoring any
    remaining section/id filters) rather than returning an empty list.
    This is the same behavior as the original module.
    """
    records: list[dict] = payload.get("records", [])
    if not records:
        logger.warning("hybrid_search: payload contains no records.")
        return []

    has_scope = any(
        v is not None
        for v in (
            source_filter,
            paper_id_filter,
            document_id_filter,
        )
    )
    effective_allow_global = (
        not has_scope
        if allow_global_search is None
        else allow_global_search
    )

    query_vector: np.ndarray = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    positions = _source_positions(
        records,
        source_filter,
        section_filter,
        paper_id_filter=paper_id_filter,
        document_id_filter=document_id_filter,
        allow_global_search=effective_allow_global,
    )
    logger.debug(
        "hybrid_search: source=%r paper_id=%r document_id=%r section=%r "
        "allow_global=%r (has_scope=%r) -> %d positions",
        source_filter, paper_id_filter, document_id_filter,
        section_filter, effective_allow_global, has_scope, len(positions),
    )

    matches = _faiss_subset_search(query_vector, index, positions, candidate_k)

    # Fallback: if source filter produced no matches at all, retry without it.
    if source_filter and not matches and effective_allow_global:
        logger.info(
            "hybrid_search: source_filter %r matched 0 positions; "
            "falling back to corpus-wide search (section/id filters retained).",
            source_filter,
        )
        fallback_positions = _source_positions(
            records,
            source_filter=None,
            section_filter=section_filter,
            paper_id_filter=paper_id_filter,
            document_id_filter=document_id_filter,
            allow_global_search=True,
        )
        matches = _faiss_subset_search(query_vector, index, fallback_positions, candidate_k)

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
                "_position": position,
            }
        )
        results.append(record)

    # Sort by fused score (fix for original sort-order bug).
    results.sort(key=lambda r: r["hybrid_score"], reverse=True)

    logger.debug(
        "hybrid_search: returning %d candidates (top hybrid=%.4f)",
        len(results),
        results[0]["hybrid_score"] if results else float("nan"),
    )
    return results


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    id_key: str = "_position",
) -> list[dict]:
    """Fuse multiple ranked candidate lists using Reciprocal Rank Fusion.

    RRF score for a document ``d``::

        RRF(d) = sum( 1 / (k + rank_L(d)) )   for each list L

    Parameters
    ----------
    ranked_lists: One list per retriever/query variant, sorted best-first.
    k:            Damping constant (default 60, Cormack et al. 2009).
    id_key:       Field used to identify the same chunk across lists.
    """
    rrf_scores: dict = defaultdict(float)
    best_record: dict = {}

    for ranked_list in ranked_lists:
        for rank, record in enumerate(ranked_list, start=1):
            doc_id = record.get(id_key)
            if doc_id is None:
                doc_id = record.get("text", "")[:200]
            rrf_scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in best_record:
                best_record[doc_id] = dict(record)

    fused = []
    for doc_id, score in rrf_scores.items():
        record = dict(best_record[doc_id])
        record["rrf_score"] = score
        fused.append(record)

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused


# ===========================================================================
# Clinical synonym expansion
# ===========================================================================

DEFAULT_CLINICAL_SYNONYMS: dict[str, list[str]] = {
    "main outcome measure": ["primary endpoint", "primary end point", "primary efficacy endpoint"],
    "primary outcome": ["primary endpoint", "primary end point"],
    "study design": ["trial design", "phase II", "phase III", "randomized", "open-label", "multicenter"],
    "drug class": ["mechanism of action", "tyrosine kinase inhibitor", "TKI", "inhibitor class"],
    "ethnicity": ["race", "nationality", "population", "European", "Asian", "Western"],
    "line of therapy": ["treatment line", "first-line", "second-line", "prior therapy"],
    "response rate": ["ORR", "objective response rate", "partial response", "complete response"],
    "progression-free survival": ["PFS", "time to progression", "progression free survival"],
    "overall survival": ["OS", "survival time"],
    "adverse events": ["toxicity", "side effects", "treatment-related", "AE", "TRAE"],
    "companion diagnostic": ["CDx", "companion test", "biomarker test", "diagnostic assay"],
    "trial registration": ["ClinicalTrials.gov", "NCT number", "trial identifier", "EudraCT"],
    "sample size": ["enrolled", "randomized", "number of patients"],
}


def expand_query_synonyms(
    query: str,
    synonym_map: Optional[dict[str, list[str]]] = None,
) -> str:
    """Append clinical-trial synonym terms to *query* for substring matches."""
    synonym_map = synonym_map if synonym_map is not None else DEFAULT_CLINICAL_SYNONYMS
    lowered = query.lower()
    additions: list[str] = []
    for term, synonyms in synonym_map.items():
        if term in lowered:
            additions.extend(synonyms)

    if not additions:
        return query

    seen: set[str] = set()
    deduped: list[str] = []
    for term in additions:
        if term.lower() not in seen:
            seen.add(term.lower())
            deduped.append(term)

    return f"{query} {' '.join(deduped)}"


# ===========================================================================
# Query classification & routing
# ===========================================================================


class QueryType(str, Enum):
    IDENTIFIER = "identifier"
    STUDY_DESIGN = "study_design"
    PRIMARY_ENDPOINT = "primary_endpoint"
    SAFETY = "safety"
    EFFICACY = "efficacy"
    ENROLLMENT = "enrollment"
    DRUG_CLASS = "drug_class"
    TEMPORAL = "temporal"
    GENERIC = "generic"


_QUERY_TYPE_RULES: list[tuple[QueryType, list[str]]] = [
    (QueryType.IDENTIFIER, ["nct number", "nct id", "clinicaltrials.gov", "eudract", "trial id", "registration number"]),
    (QueryType.PRIMARY_ENDPOINT, ["primary endpoint", "primary end point", "main outcome measure", "primary outcome"]),
    (QueryType.STUDY_DESIGN, ["study design", "trial design", "phase of the study", "randomized", "open-label", "blinded"]),
    (QueryType.SAFETY, ["adverse event", "side effect", "toxicity", "safety profile", "serious adverse"]),
    (QueryType.TEMPORAL, ["how long", "duration of", "response duration", "survival time", "time to"]),
    (QueryType.ENROLLMENT, ["how many patients", "sample size", "enrolled", "randomized", "screened"]),
    (QueryType.DRUG_CLASS, ["drug class", "class of drug", "mechanism of action", "what type of inhibitor"]),
    (QueryType.EFFICACY, ["response rate", "orr", "pfs", "progression-free", "overall survival", " os ", "efficacy"]),
]


def classify_query(query: str) -> QueryType:
    """Classify *query* into a coarse :class:`QueryType` bucket (rule-based)."""
    lowered = f" {query.lower()} "
    for query_type, triggers in _QUERY_TYPE_RULES:
        if any(trigger in lowered for trigger in triggers):
            return query_type
    return QueryType.GENERIC


@dataclass
class RouteConfig:
    """Concrete retrieval parameters resolved for a given query type."""
    query_type: QueryType
    section_filter: Optional[str] = None
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    candidate_k: int = 10
    use_multi_query: bool = False
    use_hyde: bool = False
    expand_synonyms: bool = True


_ROUTE_TABLE: dict[QueryType, RouteConfig] = {
    QueryType.IDENTIFIER: RouteConfig(
        query_type=QueryType.IDENTIFIER,
        semantic_weight=0.3, keyword_weight=0.7,
        candidate_k=15, expand_synonyms=False,
    ),
    QueryType.PRIMARY_ENDPOINT: RouteConfig(
        query_type=QueryType.PRIMARY_ENDPOINT,
        section_filter="abstract",
        semantic_weight=0.6, keyword_weight=0.4,
        candidate_k=10, expand_synonyms=True,
    ),
    QueryType.STUDY_DESIGN: RouteConfig(
        query_type=QueryType.STUDY_DESIGN,
        semantic_weight=0.5, keyword_weight=0.5,
        candidate_k=12, use_multi_query=True, expand_synonyms=True,
    ),
    QueryType.SAFETY: RouteConfig(
        query_type=QueryType.SAFETY,
        semantic_weight=0.6, keyword_weight=0.4,
        candidate_k=15, expand_synonyms=True,
    ),
    QueryType.TEMPORAL: RouteConfig(
        query_type=QueryType.TEMPORAL,
        semantic_weight=0.65, keyword_weight=0.35,
        candidate_k=15, use_multi_query=True, expand_synonyms=True,
    ),
    QueryType.ENROLLMENT: RouteConfig(
        query_type=QueryType.ENROLLMENT,
        section_filter="abstract",
        semantic_weight=0.5, keyword_weight=0.5,
        candidate_k=10, expand_synonyms=True,
    ),
    QueryType.DRUG_CLASS: RouteConfig(
        query_type=QueryType.DRUG_CLASS,
        semantic_weight=0.7, keyword_weight=0.3,
        candidate_k=10, use_hyde=True, expand_synonyms=True,
    ),
    QueryType.EFFICACY: RouteConfig(
        query_type=QueryType.EFFICACY,
        section_filter="abstract",
        semantic_weight=0.6, keyword_weight=0.4,
        candidate_k=10, expand_synonyms=True,
    ),
    QueryType.GENERIC: RouteConfig(
        query_type=QueryType.GENERIC,
        semantic_weight=0.7, keyword_weight=0.3,
        candidate_k=10, expand_synonyms=True,
    ),
}


def route_query(query: str) -> RouteConfig:
    """Classify *query* and resolve it to a concrete :class:`RouteConfig`."""
    return _ROUTE_TABLE[classify_query(query)]


# ===========================================================================
# Cross-encoder reranking
# ===========================================================================


class CrossEncoderReranker:
    """Thin wrapper around ``sentence_transformers.CrossEncoder``.

    Lazy-loaded on first :meth:`rerank` call so importing this module never
    triggers a model download unless reranking is actually used.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("CrossEncoderReranker: loading model '%s'", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: Optional[int] = None,
        text_key: str = "text",
    ) -> list[dict]:
        """Score and reorder *candidates* by cross-encoder relevance."""
        if not candidates:
            return []
        self._ensure_loaded()
        pairs = [(query, c.get(text_key, "")) for c in candidates]
        scores = self._model.predict(pairs)
        reranked = []
        for candidate, score in zip(candidates, scores):
            record = dict(candidate)
            record["rerank_score"] = float(score)
            reranked.append(record)
        reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        return reranked[:top_n] if top_n is not None else reranked


# ===========================================================================
# Pluggable LLM interface
# ===========================================================================


class LLMClient(Protocol):
    """Interface for multi-query and HyDE generation.
    Implement against any provider (OpenAI, Anthropic, local, ...).
    """
    def generate(self, prompt: str) -> str: ...


class EchoLLMClient:
    """Deterministic fallback client that returns the prompt unchanged.

    Multi-query retrieval degrades to single-query retrieval and HyDE
    retrieval degrades to plain search when an external generator is not
    supplied.
    """
    def generate(self, prompt: str) -> str:
        logger.warning(
            "EchoLLMClient.generate: no real LLM configured. "
            "Multi-query / HyDE will not expand vocabulary."
        )
        return prompt


_MULTI_QUERY_PROMPT_TEMPLATE = (
    "Generate {n} alternative phrasings of the following question "
    "about a clinical trial paper. Each phrasing should preserve the original "
    "meaning but use different terminology, abbreviations, or sentence structure "
    "that might appear in the source document. Return ONLY the {n} phrasings, "
    "one per line, with no numbering or extra commentary.\n\nQuestion: {query}\n"
)

_HYDE_PROMPT_TEMPLATE = (
    "You are writing a single hypothetical sentence that might appear "
    "in a clinical trial paper and would directly answer the following question. "
    "Write only the hypothetical sentence, in the factual, technical style of a "
    "medical journal. Do not hedge, do not mention that it is hypothetical.\n\n"
    "Question: {query}\n"
)


def _parse_llm_lines(raw: str, expected_n: int) -> list[str]:
    lines = [line.strip(" -\t") for line in raw.splitlines()]
    lines = [line for line in lines if line]
    return lines[:expected_n] if lines else []


# ===========================================================================
# Multi-query retrieval
# ===========================================================================


def multi_query_retrieve(
    query: str,
    embedder,
    index: faiss.Index,
    payload: dict,
    llm: LLMClient,
    n_variants: int = 3,
    candidate_k: int = 10,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    source_filter: Optional[str] = None,
    section_filter: Optional[str] = None,
    paper_id_filter: Optional[str] = None,
    document_id_filter: Optional[str] = None,
    allow_global_search: Optional[bool] = None,
    rrf_k: int = 60,
) -> list[dict]:
    """Retrieve for *query* plus LLM-generated reformulations, fused via RRF.

    ``allow_global_search`` defaults to ``None`` (auto-detect per query --
    see :func:`hybrid_search`); pass ``True``/``False`` to override for
    every variant.
    """
    prompt = _MULTI_QUERY_PROMPT_TEMPLATE.format(n=n_variants, query=query)
    raw = llm.generate(prompt)
    variants = _parse_llm_lines(raw, n_variants)

    if not variants:
        logger.info("multi_query_retrieve: no variants generated; using original query only.")

    ranked_lists = [
        hybrid_search(
            q, embedder, index, payload,
            candidate_k=candidate_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            source_filter=source_filter,
            section_filter=section_filter,
            paper_id_filter=paper_id_filter,
            document_id_filter=document_id_filter,
            allow_global_search=allow_global_search,
        )
        for q in [query] + variants
    ]
    return reciprocal_rank_fusion(ranked_lists, k=rrf_k)


# ===========================================================================
# HyDE
# ===========================================================================


def hyde_retrieve(
    query: str,
    embedder,
    index: faiss.Index,
    payload: dict,
    llm: LLMClient,
    candidate_k: int = 10,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    source_filter: Optional[str] = None,
    section_filter: Optional[str] = None,
    paper_id_filter: Optional[str] = None,
    document_id_filter: Optional[str] = None,
    allow_global_search: Optional[bool] = None,
    blend_with_original: bool = True,
) -> list[dict]:
    """Retrieve using a hypothetical answer embedding (HyDE).

    ``allow_global_search`` defaults to ``None`` (auto-detect -- see
    :func:`hybrid_search`).
    """
    hypothesis = llm.generate(
        _HYDE_PROMPT_TEMPLATE.format(query=query)
    ).strip() or query

    def _search(q: str) -> list[dict]:
        return hybrid_search(
            q, embedder, index, payload,
            candidate_k=candidate_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            source_filter=source_filter,
            section_filter=section_filter,
            paper_id_filter=paper_id_filter,
            document_id_filter=document_id_filter,
            allow_global_search=allow_global_search,
        )

    hyde_results = _search(hypothesis)
    if not blend_with_original:
        return hyde_results
    return reciprocal_rank_fusion([hyde_results, _search(query)])


# ===========================================================================
# Confidence-based fallback + full pipeline
# ===========================================================================


class ConfidenceLevel(str, Enum):
    HIGH = "high_confidence"
    MEDIUM = "medium_confidence"
    LOW = "low_confidence"


@dataclass
class FallbackThresholds:
    high: float = 0.85
    medium: float = 0.45


@dataclass
class RetrievalResult:
    candidates: list[dict]
    confidence: ConfidenceLevel
    query_type: QueryType
    queries_used: list[str] = field(default_factory=list)
    filters_relaxed: list[str] = field(default_factory=list)


def _top_score(candidates: list[dict]) -> float:
    if not candidates:
        return float("-inf")
    top = candidates[0]
    return top.get("rerank_score", top.get("hybrid_score", top.get("rrf_score", 0.0)))


def retrieve_with_fallback(
    query: str,
    embedder,
    index: faiss.Index,
    payload: dict,
    llm: Optional[LLMClient] = None,
    reranker: Optional[CrossEncoderReranker] = None,
    source_filter: Optional[str] = None,
    paper_id_filter: Optional[str] = None,
    document_id_filter: Optional[str] = None,
    section_filter: Optional[str] = None,
    allow_global_search: Optional[bool] = None,
    top_n: int = 5,
    thresholds: Optional[FallbackThresholds] = None,
    apply_synonym_expansion: bool = True,
) -> RetrievalResult:
    """Full production pipeline.

    classify -> route -> synonym-expand -> retrieve (multi-query|HyDE|direct)
    -> rerank -> confidence-gate -> filter-relaxation ladder -> RetrievalResult.

    ``allow_global_search`` defaults to ``None`` (auto):

    * Each retrieval attempt auto-detects whether it's scoped (based on the
      source/paper_id/document_id it's actually using at that step of the
      ladder) -- see :func:`hybrid_search`. An unscoped query always searches
      the whole corpus.
    * The final ladder rung -- dropping an explicitly-supplied
      ``source_filter`` entirely after a low-confidence scoped result -- is
      allowed unless the caller explicitly passes ``allow_global_search=False``
      to forbid it. Pass ``True`` to make that intent explicit, or ``False``
      to guarantee results never cross outside the requested document even
      as a last resort.
    """
    thresholds = thresholds or FallbackThresholds()
    route = route_query(query)
    queries_used: list[str] = [query]
    filters_relaxed: list[str] = []

    effective_query = query
    if apply_synonym_expansion and route.expand_synonyms:
        expanded = expand_query_synonyms(query)
        if expanded != query:
            effective_query = expanded
            queries_used.append(expanded)

    # Caller-supplied section_filter wins over route suggestion.
    effective_section = section_filter if section_filter is not None else route.section_filter

    def _retrieve(
        q: str,
        src: Optional[str],
        pid: Optional[str],
        did: Optional[str],
        sec: Optional[str],
    ) -> list[dict]:
        shared = dict(
            candidate_k=route.candidate_k,
            semantic_weight=route.semantic_weight,
            keyword_weight=route.keyword_weight,
            source_filter=src,
            section_filter=sec,
            paper_id_filter=pid,
            document_id_filter=did,
            allow_global_search=allow_global_search,
        )
        if route.use_multi_query:
            if llm is None:
                logger.warning(
                    "retrieve_with_fallback: multi-query requested but no LLM "
                    "supplied; falling back to direct hybrid_search."
                )
                return hybrid_search(q, embedder, index, payload, **shared)
            return multi_query_retrieve(q, embedder, index, payload, llm, **shared)

        if route.use_hyde:
            if llm is None:
                logger.warning(
                    "retrieve_with_fallback: HyDE requested but no LLM "
                    "supplied; falling back to direct hybrid_search."
                )
                return hybrid_search(q, embedder, index, payload, **shared)
            return hyde_retrieve(q, embedder, index, payload, llm, **shared)

        return hybrid_search(q, embedder, index, payload, **shared)

    candidates = _retrieve(
        effective_query, source_filter, paper_id_filter, document_id_filter, effective_section,
    )
    if reranker is not None:
        candidates = reranker.rerank(query, candidates, top_n=max(top_n, route.candidate_k))

    score = _top_score(candidates)

    # Confidence-gated filter-relaxation ladder.
    if score < thresholds.medium:
        if effective_section:
            logger.info(
                "retrieve_with_fallback: low confidence (%.4f); relaxing section_filter.", score,
            )
            filters_relaxed.append("section_filter")
            candidates = _retrieve(
                effective_query, source_filter, paper_id_filter, document_id_filter, None,
            )
            if reranker is not None:
                candidates = reranker.rerank(query, candidates, top_n=max(top_n, route.candidate_k))
            score = _top_score(candidates)

        if score < thresholds.medium and source_filter and allow_global_search is True:
            logger.info(
                "retrieve_with_fallback: still low confidence (%.4f); relaxing source_filter.", score,
            )
            filters_relaxed.append("source_filter")
            candidates = _retrieve(effective_query, None, None, None, None)
            if reranker is not None:
                candidates = reranker.rerank(query, candidates, top_n=max(top_n, route.candidate_k))
            score = _top_score(candidates)

    confidence = (
        ConfidenceLevel.HIGH if score >= thresholds.high else
        ConfidenceLevel.MEDIUM if score >= thresholds.medium else
        ConfidenceLevel.LOW
    )

    return RetrievalResult(
        candidates=candidates[:top_n],
        confidence=confidence,
        query_type=route.query_type,
        queries_used=queries_used,
        filters_relaxed=filters_relaxed,
    )

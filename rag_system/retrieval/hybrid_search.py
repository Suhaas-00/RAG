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

The original public API (``hybrid_search``, ``is_noisy``, ``keyword_score``)
is preserved with its existing call signature so existing callers keep
working. Two pre-existing gaps in the original implementation are fixed
in-place and documented inline:

1. ``hybrid_search`` previously returned candidates ordered by raw FAISS
   semantic score, *not* by the fused ``hybrid_score`` it computed. This is
   fixed by sorting explicitly before returning.
2. ``section_filter`` had no fallback path (only ``source_filter`` did).
   ``retrieve_with_fallback`` (new, see below) adds a generalised fallback
   ladder that relaxes section, then source, then both.

Public API
----------
hybrid_search(query, embedder, index, payload, ...)
    Original hybrid retrieval pipeline (semantic + keyword), now sorted by
    fused score.

is_noisy(text)
    Predicate that identifies low-quality OCR / structural text.

keyword_score(query, text)
    Normalised keyword overlap score.

reciprocal_rank_fusion(ranked_lists, k=60)
    Fuse multiple ranked candidate lists by reciprocal rank rather than raw
    score.

expand_query_synonyms(query, synonym_map=None)
    Expand a query string with clinical-trial synonym terms.

classify_query(query) / route_query(query)
    Classify a question into a retrieval-strategy bucket and resolve it to
    concrete routing parameters (section filter, weighting, k).

CrossEncoderReranker
    Thin wrapper around ``sentence_transformers.CrossEncoder`` for
    re-scoring (query, chunk) pairs.

hyde_retrieve(query, embedder, index, payload, llm, ...)
    Hypothetical Document Embeddings retrieval using a pluggable LLM
    interface.

multi_query_retrieve(query, embedder, index, payload, llm, ...)
    Generate query reformulations via a pluggable LLM, retrieve for each,
    fuse with RRF.

retrieve_with_fallback(query, embedder, index, payload, ...)
    Full production pipeline: classify -> route -> (multi-query | hyde |
    direct) -> hybrid retrieval -> RRF -> cross-encoder rerank -> confidence
    threshold -> filter-relaxation fallback ladder.

LLMClient (Protocol)
    Interface multi-query / HyDE depend on. Bring your own implementation
    (OpenAI, Anthropic, local model, etc.) by satisfying this protocol.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

import faiss
import numpy as np

from rag_system.retrieval.document_scope import filter_record_indices
from rag_system.utils.preprocessing import normalize_for_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text-quality predicates  (unchanged from original)
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
# Keyword scoring  (unchanged from original)
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
# Position helpers  (unchanged from original)
# ---------------------------------------------------------------------------


def _source_positions(
    records: list[dict],
    source_filter: Optional[str],
    section_filter: Optional[str] = None,
    paper_id_filter: Optional[str] = None,
    document_id_filter: Optional[str] = None,
    allow_global_search: bool = False,
) -> list[int]:
    """Return record indices matching the optional constraints.

    Filtering strategy
    ------------------
    1. Remove noisy records unconditionally.
    2. If *section_filter* is set, keep only matching ``section`` values.
    3. If *source_filter* is set, exact filename match first, then a
       stem-only match for legacy metadata missing the ``.pdf`` suffix.
    """
    return filter_record_indices(
        records,
        {
            "source": source_filter,
            "section": section_filter,
            "paper_id": paper_id_filter,
            "document_id": document_id_filter,
        },
        enable_document_filtering=True,
        allow_global_search=allow_global_search,
        require_document_scope=True,
        drop_noisy=True,
        noisy_predicate=is_noisy,
    )


# ---------------------------------------------------------------------------
# FAISS subset search  (unchanged from original)
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
# Public hybrid search  (BUGFIX: now sorted by fused hybrid_score)
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
    allow_global_search: bool = False,
) -> list[dict]:
    """Return the top *candidate_k* hybrid-scored chunks for *query*.

    Pipeline
    --------
    1. Encode *query* to a dense vector with *embedder*.
    2. Build an allowed-position list using *source_filter* and *section_filter*.
    3. Run FAISS inner-product search on the filtered subset.
    4. If a source filter was requested but produced zero matches, fall back
       to the whole clean corpus (retaining any section constraint).
    5. Compute per-candidate hybrid scores, sort by ``hybrid_score``
       descending, and return the enriched dicts.

    .. note::
       **Bugfix vs. original**: the original implementation returned
       candidates ordered by raw FAISS semantic score even though it
       computed a separate ``hybrid_score``. This version sorts explicitly
       by ``hybrid_score`` before returning, since that is the score the
       function's own docstring and naming promise callers.
    """
    records: list[dict] = payload.get("records", [])
    if not records:
        logger.warning("hybrid_search: payload contains no records.")
        return []

    query_vector: np.ndarray = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    positions = _source_positions(
        records,
        source_filter,
        section_filter,
        paper_id_filter=paper_id_filter,
        document_id_filter=document_id_filter,
        allow_global_search=allow_global_search,
    )
    logger.debug(
        "hybrid_search: source_filter=%r paper_id_filter=%r document_id_filter=%r section_filter=%r -> %d positions",
        source_filter,
        paper_id_filter,
        document_id_filter,
        section_filter,
        len(positions),
    )

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
                "_position": position,
            }
        )
        results.append(record)

    # BUGFIX: sort by the fused score, not FAISS's semantic-only order.
    results.sort(key=lambda r: r["hybrid_score"], reverse=True)

    logger.debug(
        "hybrid_search: returning %d candidates (top hybrid=%.4f)",
        len(results),
        results[0]["hybrid_score"] if results else float("nan"),
    )

    return results


# ===========================================================================
# NEW: Reciprocal Rank Fusion
# ===========================================================================


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    id_key: str = "_position",
) -> list[dict]:
    """Fuse multiple ranked candidate lists using Reciprocal Rank Fusion.

    RRF combines lists by *rank position* rather than raw score, which makes
    it robust to score-scale mismatches between heterogeneous retrievers
    (e.g. a BM25 list and a dense-vector list, or several reformulated-query
    result lists). This is the correct fusion strategy for combining
    :func:`multi_query_retrieve`'s per-query result lists, and is preferred
    over naive score-averaging for that purpose.

    RRF score for a document ``d`` is::

        RRF(d) = sum over lists L containing d of  1 / (k + rank_L(d))

    where ``rank_L(d)`` is the 1-indexed rank of ``d`` in list ``L``.

    Parameters
    ----------
    ranked_lists: One list of candidate dicts per retriever/query variant,
        each already sorted best-first.
    k:            RRF damping constant. Higher values flatten the
        contribution of top ranks; 60 is the standard default from the
        original RRF paper (Cormack et al., 2009) and works well without
        tuning in most settings.
    id_key:       Dict key used to identify the same underlying chunk across
        lists (defaults to the internal FAISS row id stamped by
        :func:`hybrid_search` / :func:`_faiss_subset_search`).

    Returns
    -------
    Single fused, deduplicated list of candidate dicts, sorted by descending
    RRF score, each annotated with ``rrf_score``.
    """
    rrf_scores: dict = defaultdict(float)
    best_record: dict = {}

    for ranked_list in ranked_lists:
        for rank, record in enumerate(ranked_list, start=1):
            doc_id = record.get(id_key)
            if doc_id is None:
                # Fall back to text-based identity if no stable id is present.
                doc_id = record.get("text", "")[:200]
            rrf_scores[doc_id] += 1.0 / (k + rank)
            # Keep the richest copy of the record we've seen (first wins,
            # but later copies can fill in missing fields).
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
# NEW: Clinical synonym expansion
# ===========================================================================

#: Default clinical-trial terminology map. Each key phrase, if found in the
#: query (case-insensitive substring match), has its synonym terms appended
#: to the expanded query. Extend/override via the ``synonym_map`` parameter.
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
    """Append clinical-trial synonym terms to *query* for substring matches.

    This targets the lexical-mismatch failure mode where the question uses
    plain-language phrasing ("main outcome measure") but the source document
    uses domain terminology ("primary end point"). Expanding the query
    string (rather than rewriting it) is intentionally conservative: it
    preserves the original query intent and is cheap (no LLM call), making
    it suitable to run on every query unconditionally.

    Parameters
    ----------
    query:       Raw user query.
    synonym_map: Override/replacement for :data:`DEFAULT_CLINICAL_SYNONYMS`.

    Returns
    -------
    The original query with matched synonym phrases appended, separated by
    spaces. If no terms match, the query is returned unchanged.
    """
    synonym_map = synonym_map if synonym_map is not None else DEFAULT_CLINICAL_SYNONYMS
    lowered = query.lower()
    additions: list[str] = []

    for term, synonyms in synonym_map.items():
        if term in lowered:
            additions.extend(synonyms)

    if not additions:
        return query

    # De-duplicate while preserving order.
    seen = set()
    deduped = []
    for term in additions:
        if term.lower() not in seen:
            seen.add(term.lower())
            deduped.append(term)

    return f"{query} {' '.join(deduped)}"


# ===========================================================================
# NEW: Query classification & routing
# ===========================================================================


class QueryType(str, Enum):
    """Coarse question-type buckets used to select a retrieval strategy."""

    IDENTIFIER = "identifier"          # NCT numbers, trial registration IDs
    STUDY_DESIGN = "study_design"      # phase, randomization, blinding
    PRIMARY_ENDPOINT = "primary_endpoint"
    SAFETY = "safety"                  # adverse events, toxicity
    EFFICACY = "efficacy"              # ORR, PFS, OS, response
    ENROLLMENT = "enrollment"          # sample size, screened/randomized counts
    DRUG_CLASS = "drug_class"          # mechanism, drug classification
    TEMPORAL = "temporal"              # durations, timelines, "how long"
    GENERIC = "generic"                # fallback bucket


# Ordered list of (QueryType, [trigger substrings]). Order matters: the
# first matching type wins, so more specific buckets are listed first.
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
    """Classify *query* into a coarse :class:`QueryType` bucket.

    This is a fast, dependency-free, rule-based classifier intended as the
    default routing mechanism. It deliberately avoids requiring an LLM call
    on every query, since classification only needs to choose among a small,
    fixed set of retrieval strategies (see :func:`route_query`).

    For higher accuracy on ambiguous phrasing, swap this out for an
    LLM-based or fine-tuned classifier behind the same signature
    (``str -> QueryType``); :func:`route_query` is agnostic to how the type
    was produced.
    """
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


# Routing table: how each QueryType should be retrieved. Tune per corpus.
_ROUTE_TABLE: dict[QueryType, RouteConfig] = {
    QueryType.IDENTIFIER: RouteConfig(
        query_type=QueryType.IDENTIFIER,
        section_filter=None,
        semantic_weight=0.3,
        keyword_weight=0.7,   # identifiers are exact strings; favor lexical match
        candidate_k=15,
        expand_synonyms=False,
    ),
    QueryType.PRIMARY_ENDPOINT: RouteConfig(
        query_type=QueryType.PRIMARY_ENDPOINT,
        section_filter="abstract",
        semantic_weight=0.6,
        keyword_weight=0.4,
        candidate_k=10,
        expand_synonyms=True,
    ),
    QueryType.STUDY_DESIGN: RouteConfig(
        query_type=QueryType.STUDY_DESIGN,
        section_filter=None,
        semantic_weight=0.5,
        keyword_weight=0.5,
        candidate_k=12,
        use_multi_query=True,
        expand_synonyms=True,
    ),
    QueryType.SAFETY: RouteConfig(
        query_type=QueryType.SAFETY,
        section_filter=None,
        semantic_weight=0.6,
        keyword_weight=0.4,
        candidate_k=15,
        expand_synonyms=True,
    ),
    QueryType.TEMPORAL: RouteConfig(
        query_type=QueryType.TEMPORAL,
        section_filter=None,
        semantic_weight=0.65,
        keyword_weight=0.35,
        candidate_k=15,
        use_multi_query=True,
        expand_synonyms=True,
    ),
    QueryType.ENROLLMENT: RouteConfig(
        query_type=QueryType.ENROLLMENT,
        section_filter="abstract",
        semantic_weight=0.5,
        keyword_weight=0.5,
        candidate_k=10,
        expand_synonyms=True,
    ),
    QueryType.DRUG_CLASS: RouteConfig(
        query_type=QueryType.DRUG_CLASS,
        section_filter=None,
        semantic_weight=0.7,
        keyword_weight=0.3,
        candidate_k=10,
        use_hyde=True,         # taxonomy questions benefit from a hypothesized answer
        expand_synonyms=True,
    ),
    QueryType.EFFICACY: RouteConfig(
        query_type=QueryType.EFFICACY,
        section_filter="abstract",
        semantic_weight=0.6,
        keyword_weight=0.4,
        candidate_k=10,
        expand_synonyms=True,
    ),
    QueryType.GENERIC: RouteConfig(
        query_type=QueryType.GENERIC,
        section_filter=None,
        semantic_weight=0.7,
        keyword_weight=0.3,
        candidate_k=10,
        expand_synonyms=True,
    ),
}


def route_query(query: str) -> RouteConfig:
    """Classify *query* and resolve it to a concrete :class:`RouteConfig`."""
    query_type = classify_query(query)
    return _ROUTE_TABLE[query_type]


# ===========================================================================
# NEW: Cross-encoder reranking
# ===========================================================================


class CrossEncoderReranker:
    """Thin wrapper around ``sentence_transformers.CrossEncoder``.

    Cross-encoders score a (query, passage) pair jointly through a single
    transformer forward pass, which is substantially more accurate than
    comparing independently-computed embeddings (bi-encoder / dense
    retrieval), at the cost of being too slow to run over an entire corpus.
    The standard pattern -- used here -- is therefore: retrieve a generous
    candidate set cheaply (FAISS + keyword), then rerank only that small
    candidate set with the cross-encoder.

    The model is loaded lazily on first use so importing this module never
    requires a network call or GPU unless reranking is actually invoked.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy-loaded

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # local import: optional dep

            logger.info("CrossEncoderReranker: loading model '%s'", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: Optional[int] = None,
        text_key: str = "text",
    ) -> list[dict]:
        """Score and reorder *candidates* by cross-encoder relevance to *query*.

        Parameters
        ----------
        query:      The user query.
        candidates: Candidate chunk dicts (e.g. output of :func:`hybrid_search`
            or :func:`reciprocal_rank_fusion`).
        top_n:      If given, truncate to the top N after reranking.
        text_key:   Dict key holding the chunk's text content.

        Returns
        -------
        Candidates sorted by descending ``rerank_score``, each annotated
        with that field. If *candidates* is empty, returns ``[]`` without
        loading the model.
        """
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
# NEW: Pluggable LLM interface (for multi-query and HyDE)
# ===========================================================================


class LLMClient(Protocol):
    """Minimal interface required for multi-query and HyDE generation.

    Implement this against whatever LLM provider you use (OpenAI, Anthropic,
    a local model, etc.). Only a single method is required, keeping the
    retrieval code provider-agnostic.
    """

    def generate(self, prompt: str) -> str:
        """Return a single text completion for *prompt*."""
        ...


class EchoLLMClient:
    """No-op :class:`LLMClient` used as a safe default / testing stub.

    ``generate`` simply returns its input unchanged, so callers can plug
    this in before wiring up a real provider and still exercise the full
    retrieval pipeline end-to-end (multi-query degrades to a single query;
    HyDE degrades to embedding the raw query). Replace with a real client
    for production use -- this stub will not actually expand vocabulary.
    """

    def generate(self, prompt: str) -> str:
        logger.warning(
            "EchoLLMClient.generate called -- no real LLM is configured. "
            "Multi-query / HyDE will not meaningfully improve recall until "
            "a real LLMClient implementation is supplied."
        )
        return prompt


_MULTI_QUERY_PROMPT_TEMPLATE = """Generate {n} alternative phrasings of the following question \
about a clinical trial paper. Each phrasing should preserve the original \
meaning but use different terminology, abbreviations, or sentence structure \
that might appear in the source document. Return ONLY the {n} phrasings, \
one per line, with no numbering or extra commentary.

Question: {query}
"""

_HYDE_PROMPT_TEMPLATE = """You are writing a single hypothetical sentence that might appear \
in a clinical trial paper and would directly answer the following question. \
Write only the hypothetical sentence, in the factual, technical style of a \
medical journal. Do not hedge, do not mention that it is hypothetical.

Question: {query}
"""


def _parse_llm_lines(raw: str, expected_n: int) -> list[str]:
    """Split an LLM completion into up to *expected_n* non-empty lines."""
    lines = [line.strip(" -\t") for line in raw.splitlines()]
    lines = [line for line in lines if line]
    return lines[:expected_n] if lines else []


# ===========================================================================
# NEW: Multi-query retrieval
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
    allow_global_search: bool = False,
    rrf_k: int = 60,
) -> list[dict]:
    """Retrieve using the original query plus *n_variants* LLM reformulations.

    Each variant is retrieved independently via :func:`hybrid_search`
    (sharing the same source/section constraints), and the resulting ranked
    lists are combined with :func:`reciprocal_rank_fusion`. RRF (rather than
    score-averaging) is used deliberately, since hybrid scores from
    different query phrasings are not directly comparable in magnitude.

    This targets the vocabulary-gap failure mode: a query phrased as
    "what was the main outcome measure" may retrieve poorly against a
    document that says "primary end point", but a reformulation generated
    by the LLM may bridge that gap.

    Parameters
    ----------
    llm:         Any :class:`LLMClient` implementation. Pass
        :class:`EchoLLMClient` to run this function without a real LLM
        configured (it will degrade to single-query retrieval).
    n_variants:  Number of alternative phrasings to generate, in addition to
        the original query.
    rrf_k:       Damping constant forwarded to :func:`reciprocal_rank_fusion`.

    Returns
    -------
    Single fused ranked list of candidate dicts.
    """
    prompt = _MULTI_QUERY_PROMPT_TEMPLATE.format(n=n_variants, query=query)
    raw_completion = llm.generate(prompt)
    variants = _parse_llm_lines(raw_completion, n_variants)

    if not variants:
        logger.info("multi_query_retrieve: no variants generated, using original query only.")

    all_queries = [query] + variants
    logger.debug("multi_query_retrieve: querying with %d variants total", len(all_queries))

    ranked_lists = [
        hybrid_search(
            q,
            embedder,
            index,
            payload,
            candidate_k=candidate_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            source_filter=source_filter,
            section_filter=section_filter,
            paper_id_filter=paper_id_filter,
            document_id_filter=document_id_filter,
            allow_global_search=allow_global_search,
        )
        for q in all_queries
    ]

    return reciprocal_rank_fusion(ranked_lists, k=rrf_k)


# ===========================================================================
# NEW: HyDE (Hypothetical Document Embeddings)
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
    allow_global_search: bool = False,
    blend_with_original: bool = True,
) -> list[dict]:
    """Retrieve using an LLM-generated hypothetical answer instead of the raw query.

    HyDE addresses queries where the *question's* vocabulary is far from the
    *answer's* vocabulary in embedding space (e.g. "which ethnicity does the
    patient belong to" vs. a document that says "European patients"). The
    LLM is asked to hallucinate a plausible answer sentence; that sentence,
    being phrased like the target document, embeds closer to the true
    answer chunk than the bare question does.

    Parameters
    ----------
    llm:                  Any :class:`LLMClient` implementation. With
        :class:`EchoLLMClient`, this degrades to plain ``hybrid_search`` on
        the original query.
    blend_with_original:  If ``True`` (default), retrieve with both the
        original query and the hypothetical document, then fuse with RRF.
        This hedges against a poor/hallucinated hypothesis dominating
        retrieval. If ``False``, only the hypothesis is used.

    Returns
    -------
    Ranked list of candidate dicts.
    """
    prompt = _HYDE_PROMPT_TEMPLATE.format(query=query)
    hypothesis = llm.generate(prompt).strip()

    if not hypothesis:
        logger.info("hyde_retrieve: empty hypothesis generated, falling back to raw query.")
        hypothesis = query

    hyde_results = hybrid_search(
        hypothesis,
        embedder,
        index,
        payload,
        candidate_k=candidate_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        source_filter=source_filter,
        section_filter=section_filter,
        paper_id_filter=paper_id_filter,
        document_id_filter=document_id_filter,
        allow_global_search=allow_global_search,
    )

    if not blend_with_original:
        return hyde_results

    original_results = hybrid_search(
        query,
        embedder,
        index,
        payload,
        candidate_k=candidate_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        source_filter=source_filter,
        section_filter=section_filter,
        paper_id_filter=paper_id_filter,
        document_id_filter=document_id_filter,
        allow_global_search=allow_global_search,
    )

    return reciprocal_rank_fusion([hyde_results, original_results])


# ===========================================================================
# NEW: Confidence-based fallback thresholds + filter-relaxation ladder
# ===========================================================================


class ConfidenceLevel(str, Enum):
    HIGH = "high_confidence"
    MEDIUM = "medium_confidence"
    LOW = "low_confidence"


@dataclass
class FallbackThresholds:
    """Score thresholds (on the post-rerank ``rerank_score``, if available,
    else ``hybrid_score``) that determine confidence routing."""

    high: float = 0.85
    medium: float = 0.45


@dataclass
class RetrievalResult:
    """Final structured output of :func:`retrieve_with_fallback`."""

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
    allow_global_search: bool = False,
    top_n: int = 5,
    thresholds: Optional[FallbackThresholds] = None,
    apply_synonym_expansion: bool = True,
) -> RetrievalResult:
    """End-to-end production retrieval: classify -> route -> retrieve ->
    fuse -> rerank -> confidence-gate -> relax filters if needed.

    Pipeline
    --------
    1. Classify the query (:func:`classify_query`) and resolve routing
       parameters (:func:`route_query`).
    2. Optionally expand the query with clinical synonyms
       (:func:`expand_query_synonyms`), unless the route disables it
       (e.g. identifier queries, where expansion would dilute exact-match
       lexical scoring).
    3. Retrieve via the strategy the route specifies: multi-query, HyDE, or
       direct :func:`hybrid_search`.
    4. If a :class:`CrossEncoderReranker` is supplied, rerank the candidate
       set.
    5. Compare the top score against *thresholds*:
         * score >= thresholds.high   -> return as HIGH confidence
         * score >= thresholds.medium -> return as MEDIUM confidence
         * otherwise                  -> relax filters and retry:
               a. drop section_filter (if set), retry
               b. drop source_filter (if set), retry
               c. return whatever was obtained as LOW confidence
    6. Truncate to *top_n* and return a :class:`RetrievalResult`.

    Parameters
    ----------
    llm:       Required only if the resolved route requests multi-query or
        HyDE retrieval. If ``None`` and the route requests either, this
        function logs a warning and falls back to direct
        :func:`hybrid_search` for that query rather than raising, so callers
        without an LLM configured still get a working (if less recall-
        optimized) pipeline.
    reranker:  Optional :class:`CrossEncoderReranker`. If omitted, ranking
        falls back to the fused ``hybrid_score`` / ``rrf_score`` and
        confidence thresholds are interpreted against that score instead.
    thresholds: Override default :class:`FallbackThresholds`. Note that
        cross-encoder scores and hybrid scores are on different scales --
        recalibrate thresholds if you change whether a reranker is used.

    Returns
    -------
    :class:`RetrievalResult` with the final candidate list, the confidence
    level reached, the query type used for routing, every query string
    actually issued, and a record of which filters were relaxed (useful for
    logging/debugging retrieval quality in production).
    """
    thresholds = thresholds or FallbackThresholds()
    route = route_query(query)
    queries_used = [query]
    filters_relaxed: list[str] = []

    effective_query = query
    if apply_synonym_expansion and route.expand_synonyms:
        effective_query = expand_query_synonyms(query)
        if effective_query != query:
            queries_used.append(effective_query)

    # Route's section_filter is a *default suggestion*; an explicit caller
    # value always wins.
    effective_section_filter = section_filter if section_filter is not None else route.section_filter

    def _retrieve(
        q: str,
        src: Optional[str],
        pid: Optional[str],
        did: Optional[str],
        sec: Optional[str],
    ) -> list[dict]:
        if route.use_multi_query:
            if llm is None:
                logger.warning(
                    "retrieve_with_fallback: route requested multi-query but no "
                    "LLMClient was supplied; using direct hybrid_search instead."
                )
                return hybrid_search(
                    q, embedder, index, payload,
                    candidate_k=route.candidate_k,
                    semantic_weight=route.semantic_weight,
                    keyword_weight=route.keyword_weight,
                    source_filter=src, section_filter=sec,
                    paper_id_filter=pid, document_id_filter=did,
                    allow_global_search=allow_global_search,
                )
            return multi_query_retrieve(
                q, embedder, index, payload, llm,
                candidate_k=route.candidate_k,
                semantic_weight=route.semantic_weight,
                keyword_weight=route.keyword_weight,
                source_filter=src, section_filter=sec,
                paper_id_filter=pid, document_id_filter=did,
                allow_global_search=allow_global_search,
            )
        if route.use_hyde:
            if llm is None:
                logger.warning(
                    "retrieve_with_fallback: route requested HyDE but no "
                    "LLMClient was supplied; using direct hybrid_search instead."
                )
                return hybrid_search(
                    q, embedder, index, payload,
                    candidate_k=route.candidate_k,
                    semantic_weight=route.semantic_weight,
                    keyword_weight=route.keyword_weight,
                    source_filter=src, section_filter=sec,
                    paper_id_filter=pid, document_id_filter=did,
                    allow_global_search=allow_global_search,
                )
            return hyde_retrieve(
                q, embedder, index, payload, llm,
                candidate_k=route.candidate_k,
                semantic_weight=route.semantic_weight,
                keyword_weight=route.keyword_weight,
                source_filter=src, section_filter=sec,
                paper_id_filter=pid, document_id_filter=did,
                allow_global_search=allow_global_search,
            )
        return hybrid_search(
            q, embedder, index, payload,
            candidate_k=route.candidate_k,
            semantic_weight=route.semantic_weight,
            keyword_weight=route.keyword_weight,
            source_filter=src, section_filter=sec,
            paper_id_filter=pid, document_id_filter=did,
            allow_global_search=allow_global_search,
        )

    candidates = _retrieve(
        effective_query,
        source_filter,
        paper_id_filter,
        document_id_filter,
        effective_section_filter,
    )
    if reranker is not None:
        candidates = reranker.rerank(query, candidates, top_n=max(top_n, route.candidate_k))

    score = _top_score(candidates)

    # --- Confidence-gated fallback ladder -------------------------------
    if score < thresholds.medium:
        if effective_section_filter:
            logger.info(
                "retrieve_with_fallback: low confidence (%.4f); relaxing section_filter.",
                score,
            )
            filters_relaxed.append("section_filter")
            candidates = _retrieve(
                effective_query,
                source_filter,
                paper_id_filter,
                document_id_filter,
                None,
            )
            if reranker is not None:
                candidates = reranker.rerank(query, candidates, top_n=max(top_n, route.candidate_k))
            score = _top_score(candidates)

        if score < thresholds.medium and source_filter and allow_global_search:
            logger.info(
                "retrieve_with_fallback: still low confidence (%.4f); relaxing source_filter.",
                score,
            )
            filters_relaxed.append("source_filter")
            candidates = _retrieve(effective_query, None, None, None, None)
            if reranker is not None:
                candidates = reranker.rerank(query, candidates, top_n=max(top_n, route.candidate_k))
            score = _top_score(candidates)

    if score >= thresholds.high:
        confidence = ConfidenceLevel.HIGH
    elif score >= thresholds.medium:
        confidence = ConfidenceLevel.MEDIUM
    else:
        confidence = ConfidenceLevel.LOW

    return RetrievalResult(
        candidates=candidates[:top_n],
        confidence=confidence,
        query_type=route.query_type,
        queries_used=queries_used,
        filters_relaxed=filters_relaxed,
    )

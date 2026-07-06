"""Hybrid retriever: FAISS dense search + TF-IDF keyword re-scoring + metadata filtering.

The :class:`RAGRetriever` loads the serialised payload produced by
:mod:`rag_system.ingestion.DataIngestion` and exposes a single :meth:`retrieve`
method that:

1. **Metadata filter** (pre-retrieval)
   Narrows the candidate pool to chunks whose ``source`` or structured
   ``metadata`` fields match any supplied filters before FAISS search.
   Supported filters:

   - ``source_filter``     - exact or substring match on the PDF filename.
   - ``section_filter``    - exact match on the normalised section label.
   - ``disease_filter``    - substring match against the chunk's metadata
                             diseases list.
   - ``gene_filter``       - substring match against the chunk's metadata
                             genes list.
   - ``year_filter``       - exact match on the publication year string.
   - ``chunk_type_filter`` - "content", "metadata", or None for both.
   - ``paper_id_filter`` / ``document_id_filter`` - exact ID match.

2. **Dense retrieval** (FAISS IndexFlatIP)
   Scores all filtered candidates by cosine similarity to the query vector.

3. **Keyword re-scoring** (TF-IDF dot product)
   Adds a weighted TF-IDF score to the dense score to lift exact-keyword
   matches that the embedding model may under-rank.

4. **Hybrid ranking**
   ``final_score = alpha * dense_score + (1 - alpha) * keyword_score``
   where alpha is configurable (default 0.7).

5. **Context assembly**
   Top-k chunks are formatted as ``[Source: ... | Section: ... | Page: ...]\\n<text>``
   blocks joined by ``\\n\\n``.

Document-scope integration (``document_scope.py``)
---------------------------------------------------
``document_scope`` is **not** imported at module load time and is **not**
required. By default this module filters records with a self-contained
inline implementation (source/section/disease/gene/year/chunk_type/
paper_id/document_id, plus noise dropping) so it works standalone even if
``document_scope.py`` doesn't exist in your deployment.

To opt in to the richer ``document_scope.filter_record_indices`` pipeline
(compiled filters, multi-alias support, etc.) at any point:

    from rag_system.retrieval.document_scope import filter_record_indices
    from rag_system.retrieval import rag_retriever as rr
    rr.enable_document_scope(filter_record_indices)

This is a process-level switch; every :class:`RAGRetriever` instance created
afterwards (and any already-created instance) will use it automatically.
Call ``rr.disable_document_scope()`` to revert to the inline filter.

Public API
----------
RetrievalResult(context, chunks)            - Named result container.
RAGRetriever.load(index_dir)                - Load from serialised payload.
RAGRetriever.retrieve(query, ...)           - Hybrid retrieval with metadata filtering.
RAGRetriever.retrieve_by_metadata(filters)  - Pure metadata-based retrieval
                                               (no FAISS, no TF-IDF).
enable_document_scope(filter_fn)            - Opt in to the external filter.
disable_document_scope()                    - Revert to the inline filter.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import faiss
import numpy as np

from rag_system.ingestion.embedding import PubMedEmbedder
from rag_system.retrieval.retrieval_config import document_scope_enabled as config_document_scope_enabled
from rag_system.utils.preprocessing import normalize_for_embedding

logger = logging.getLogger(__name__)

# Minimum schema version this retriever understands.
_MIN_SCHEMA_VERSION: int = 1
# Weight for dense score in hybrid ranking.
_DEFAULT_ALPHA: float = 0.7
# Score floor - chunks below this combined score are dropped.
_SCORE_FLOOR: float = 0.0


# ---------------------------------------------------------------------------
# Optional document_scope hook
# ---------------------------------------------------------------------------
# Default: None. document_scope.py is NEVER imported by this module unless
# the caller explicitly opts in via enable_document_scope(). Until then,
# _apply_metadata_filters() below uses its own inline filtering logic.

_external_filter_fn: Optional[Callable[..., list[int]]] = None


def enable_document_scope(filter_fn: Callable[..., list[int]]) -> None:
    """Swap the inline metadata filter for an external implementation.

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
        from rag_system.retrieval import rag_retriever as rr
        rr.enable_document_scope(filter_record_indices)

    This is a process-level switch -- call once at startup; every
    :class:`RAGRetriever` instance will use the supplied function from then
    on, since instances look up the module-level hook at call time rather
    than caching it at construction time.
    """
    global _external_filter_fn
    _external_filter_fn = filter_fn
    logger.info(
        "rag_retriever: document_scope filtering enabled via %s",
        getattr(filter_fn, "__qualname__", repr(filter_fn)),
    )


def disable_document_scope() -> None:
    """Revert to the built-in inline filtering (useful in tests)."""
    global _external_filter_fn
    _external_filter_fn = None
    logger.info("rag_retriever: document_scope filtering disabled; using inline filter.")


def document_scope_enabled() -> bool:
    """Return ``True`` if an external document_scope filter is active."""
    return _external_filter_fn is not None


# ---------------------------------------------------------------------------
# Text-quality predicate (self-contained; mirrors hybrid_search.is_noisy)
# ---------------------------------------------------------------------------

_NOISE_MARKERS: frozenset[str] = frozenset(
    {"table", "figure", "downloaded from", "references", "supplementary"}
)
_MAX_DIGIT_RATIO: float = 0.30


def is_noisy(text: Optional[str]) -> bool:
    """Return ``True`` for chunks that are structural noise rather than prose."""
    lowered = (text or "").lower()
    if any(marker in lowered for marker in _NOISE_MARKERS):
        return True
    length = max(len(lowered), 1)
    return sum(ch.isdigit() for ch in lowered) / length > _MAX_DIGIT_RATIO


# ---------------------------------------------------------------------------
# Inline metadata filter (default implementation; no external dependency)
# ---------------------------------------------------------------------------

_DOCUMENT_ID_KEYS = ("paper_id", "document_id", "doc_id", "source_id")
_SOURCE_KEYS = ("source", "filename", "file_name")


def _record_field(record: dict, key: str):
    """Look up *key* on a record, falling back to a nested ``metadata`` dict."""
    value = record.get(key)
    if value is None and isinstance(record.get("metadata"), dict):
        value = record["metadata"].get(key)
    return value


def _source_matches(record: dict, wanted: str) -> bool:
    """Substring-or-exact match on filename fields (handles missing '.pdf')."""
    wanted_lower = wanted.casefold()
    wanted_path = Path(wanted)
    wanted_name = wanted_path.name.casefold()
    wanted_stem = wanted_path.stem.casefold()
    for key in _SOURCE_KEYS:
        actual = record.get(key)
        if actual is None:
            continue
        actual_str = str(actual)
        actual_path = Path(actual_str)
        if actual_path.name.casefold() == wanted_name:
            return True
        if actual_path.stem.casefold() == wanted_stem:
            return True
        if wanted_lower in actual_str.casefold():
            return True
    return False


def _id_matches(record: dict, wanted: str) -> bool:
    wanted_fold = wanted.casefold()
    for key in _DOCUMENT_ID_KEYS:
        value = _record_field(record, key)
        if value is not None and str(value).casefold() == wanted_fold:
            return True
    return False


def _list_field_contains(record: dict, list_key: str, wanted: str) -> bool:
    """Substring match of *wanted* against a metadata list field (diseases/genes)."""
    metadata = record.get("metadata")
    values = None
    if isinstance(metadata, dict):
        values = metadata.get(list_key)
    if values is None:
        values = record.get(list_key)
    if not values:
        return False
    wanted_fold = wanted.casefold()
    if isinstance(values, str):
        values = [values]
    return any(wanted_fold in str(v).casefold() for v in values)


def _year_matches(record: dict, wanted: str) -> bool:
    value = _record_field(record, "year")
    return value is not None and str(value) == str(wanted)


def _inline_filter_record_indices(
    records: list[dict],
    filters: dict,
    *,
    enable_document_filtering: bool = True,
    allow_global_search: bool = False,
    require_document_scope: bool = True,
    drop_noisy: bool = True,
    noisy_predicate: Optional[Callable[[Optional[str]], bool]] = None,
) -> list[int]:
    """Default, dependency-free implementation of ``filter_record_indices``.

    Supports the same filter keys as ``document_scope.filter_record_indices``
    that this codebase relies on: ``source``, ``section``, ``disease``,
    ``gene``, ``year``, ``chunk_type``, ``paper_id``, ``document_id``.

    If ``enable_document_filtering`` is ``False``, document identity filters
    are ignored while non-document metadata filters still apply.

    If any filter is active, produces zero matches, and
    ``allow_global_search`` is ``False`` (and ``require_document_scope`` is
    ``True``), returns an empty list rather than silently falling back to
    the full corpus.
    """
    noisy_predicate = noisy_predicate or is_noisy

    source_filter = filters.get("source")
    section_filter = filters.get("section")
    disease_filter = filters.get("disease")
    gene_filter = filters.get("gene")
    year_filter = filters.get("year")
    chunk_type_filter = filters.get("chunk_type")
    paper_id_filter = filters.get("paper_id")
    document_id_filter = filters.get("document_id")
    if not enable_document_filtering:
        source_filter = None
        paper_id_filter = None
        document_id_filter = None

    result: list[int] = []
    for idx, record in enumerate(records):
        if drop_noisy and noisy_predicate(record.get("text", "")):
            continue
        if section_filter and str(record.get("section", "")).casefold() != str(section_filter).casefold():
            continue
        if chunk_type_filter and str(record.get("chunk_type", "content")).casefold() != str(chunk_type_filter).casefold():
            continue
        if disease_filter and not _list_field_contains(record, "diseases", str(disease_filter)):
            continue
        if gene_filter and not _list_field_contains(record, "genes", str(gene_filter)):
            continue
        if year_filter and not _year_matches(record, str(year_filter)):
            continue
        if paper_id_filter and not _id_matches(record, str(paper_id_filter)):
            continue
        if document_id_filter and not _id_matches(record, str(document_id_filter)):
            continue
        if source_filter and not _source_matches(record, str(source_filter)):
            continue
        result.append(idx)

    any_filter_active = any(
        v is not None
        for v in (
            source_filter, section_filter, disease_filter, gene_filter,
            year_filter, chunk_type_filter, paper_id_filter, document_id_filter,
        )
    )
    if (
        any_filter_active
        and not result
        and require_document_scope
        and not allow_global_search
    ):
        logger.debug(
            "_inline_filter_record_indices: filters produced 0 matches %s "
            "(allow_global_search=%r)",
            filters, allow_global_search,
        )
        return []

    return result


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Container returned by :meth:`RAGRetriever.retrieve`.

    Attributes
    ----------
    context:
        Pre-formatted context string ready to be injected into an LLM prompt.
    chunks:
        Ordered list of raw chunk dicts (highest score first) with an
        additional ``final_score`` key added by the retriever.
    """
    context: str
    chunks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class RAGRetriever:
    """Hybrid retriever backed by a FAISS index and TF-IDF matrix.

    Do not construct directly -- use :meth:`load`.

    Parameters
    ----------
    index:      Loaded FAISS ``IndexFlatIP`` instance.
    payload:    Full metadata pickle dict produced by the ingestion pipeline.
    embedder:   :class:`~rag_system.ingestion.embedding.PubMedEmbedder` instance
                using the **same** model as ingestion.
    alpha:      Dense-score weight in hybrid ranking ``[0, 1]``.
                ``alpha=1`` -> pure dense; ``alpha=0`` -> pure keyword.
    enable_document_filtering:
                Master on/off switch for metadata filtering (source, section,
                disease, gene, year, ids). When ``False``, filtering is
                skipped entirely and every call searches the full corpus
                (minus noisy chunks).
    allow_global_search:
                Fallback used when a query is *scoped* (a source/paper_id/
                document_id filter was supplied) but that filter matches
                nothing. Overridable per-call via the ``allow_global_search``
                argument on :meth:`retrieve` / :meth:`retrieve_by_metadata`.

                This does **not** affect unscoped queries -- a query with no
                source/paper_id/document_id filter is always allowed to
                search the full corpus automatically, regardless of this
                setting, unless a caller explicitly passes
                ``allow_global_search=False`` for that call. See
                :meth:`_apply_metadata_filters` for the exact rule.
    """

    def __init__(
        self,
        index: faiss.Index,
        payload: dict,
        embedder: PubMedEmbedder,
        *,
        alpha: float = _DEFAULT_ALPHA,
        enable_document_filtering: bool | None = None,
        allow_global_search: bool = False,
    ) -> None:
        self.index = index
        self.payload = payload
        self.embedder = embedder
        self.alpha = alpha
        self.enable_document_filtering = config_document_scope_enabled(enable_document_filtering)
        self.allow_global_search = allow_global_search

        self._records: list[dict] = payload["records"]
        self._id_to_pos: dict[str, int] = payload["id_to_position"]
        self._tfidf = payload["tfidf_vectorizer"]
        self._tfidf_matrix = payload["tfidf_matrix"]
        self._doc_metadata_index: dict[str, dict] = payload.get("doc_metadata_index", {})

        logger.info(
            "RAGRetriever ready -- %d records, index.ntotal=%d, document_scope=%s",
            len(self._records),
            self.index.ntotal,
            "external" if document_scope_enabled() else "inline",
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        *,
        alpha: float = _DEFAULT_ALPHA,
        enable_document_filtering: bool | None = None,
        allow_global_search: bool = False,
    ) -> "RAGRetriever":
        """Load a retriever from the serialised payload in *index_dir*.

        Parameters
        ----------
        index_dir:
            Directory containing ``vectors.index`` and ``metadata.pkl``.
        alpha:
            Dense-score weight for hybrid ranking.

        Raises
        ------
        FileNotFoundError
            If the index or metadata files are missing.
        ValueError
            If the payload schema version is incompatible.
        """
        index_dir = Path(index_dir)
        vectors_path = index_dir / "vectors.index"
        metadata_path = index_dir / "metadata.pkl"

        for p in (vectors_path, metadata_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        logger.info("Loading FAISS index from '%s'", vectors_path)
        index = faiss.read_index(str(vectors_path))

        logger.info("Loading metadata payload from '%s'", metadata_path)
        with metadata_path.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301

        schema_v = payload.get("schema_version", 0)
        if schema_v < _MIN_SCHEMA_VERSION:
            raise ValueError(
                f"Payload schema version {schema_v} is too old "
                f"(minimum supported: {_MIN_SCHEMA_VERSION}). "
                "Re-run ingestion to rebuild the index."
            )

        embedder = PubMedEmbedder(payload["model_name"], use_prefixes=True)
        return cls(
            index,
            payload,
            embedder,
            alpha=alpha,
            enable_document_filtering=enable_document_filtering,
            allow_global_search=allow_global_search,
        )

    # ------------------------------------------------------------------
    # Metadata filtering helpers
    # ------------------------------------------------------------------

    def _apply_metadata_filters(
        self,
        source_filter: Optional[str],
        section_filter: Optional[str],
        disease_filter: Optional[str],
        gene_filter: Optional[str],
        year_filter: Optional[str],
        chunk_type_filter: Optional[str],
        paper_id_filter: Optional[str] = None,
        document_id_filter: Optional[str] = None,
        allow_global_search: Optional[bool] = None,
    ) -> list[int]:
        """Return FAISS row indices that pass document-scope and metadata filters.

        Uses the external ``document_scope.filter_record_indices`` if
        :func:`enable_document_scope` has been called; otherwise falls back
        to the module's self-contained inline filter. Either way, filtering
        is only performed when a filter is actually supplied or
        ``self.enable_document_filtering`` is on -- callers with no filters
        pay no extra cost beyond noise-dropping.

        Global-search auto-detection
        -----------------------------
        A document is considered "in scope" only when ``source_filter``,
        ``paper_id_filter``, or ``document_id_filter`` is supplied -- these
        are the identifiers ``document_scope.py`` treats as document scope.

        * If the caller passes an explicit ``allow_global_search`` (``True``
          or ``False``), that value always wins.
        * If the caller leaves it as ``None`` (the default) and the query is
          scoped, we fall back to the instance-level ``self.allow_global_search``
          (default ``False`` -- a scoped query that matches nothing stays
          empty rather than silently searching everything).
        * If the caller leaves it as ``None`` and the query is *unscoped*
          (no document identifier at all), global search is allowed
          automatically. A user without a specific paper in mind should
          still get an answer from the whole corpus rather than an empty
          result caused purely by the document-scope guard.
        """
        has_scope = any(
            v is not None for v in (source_filter, paper_id_filter, document_id_filter)
        )
        if allow_global_search is None:
            effective_allow_global = True if not has_scope else self.allow_global_search
        else:
            effective_allow_global = allow_global_search
        filters = {
            "source": source_filter,
            "section": section_filter,
            "disease": disease_filter,
            "gene": gene_filter,
            "year": year_filter,
            "chunk_type": chunk_type_filter,
            "paper_id": paper_id_filter,
            "document_id": document_id_filter,
        }
        filters = {key: value for key, value in filters.items() if value is not None}

        filter_fn = _external_filter_fn or _inline_filter_record_indices
        indices = filter_fn(
            self._records,
            filters,
            enable_document_filtering=self.enable_document_filtering,
            allow_global_search=effective_allow_global,
            require_document_scope=not effective_allow_global,
            drop_noisy=True,
            noisy_predicate=is_noisy,
        )
        logger.debug(
            "_apply_metadata_filters: %d/%d records pass filters %s (source=%s)",
            len(indices),
            len(self._records),
            filters,
            "external" if _external_filter_fn is not None else "inline",
        )
        return indices

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
        disease_filter: Optional[str] = None,
        gene_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
        paper_id_filter: Optional[str] = None,
        document_id_filter: Optional[str] = None,
        allow_global_search: Optional[bool] = None,
        alpha: Optional[float] = None,
    ) -> RetrievalResult:
        """Retrieve the *top_k* most relevant chunks for *query*.

        Parameters
        ----------
        query:
            Raw natural-language question or keyword string.
        top_k:
            Maximum number of chunks to return.
        source_filter:
            Substring matched against the PDF source filename
            (e.g. ``"27022036"`` matches ``"27022036.pdf"``).
        section_filter:
            Exact section name (``"abstract"``, ``"methods"``, etc.).
        disease_filter:
            Substring matched against the document's extracted disease list.
        gene_filter:
            Substring matched against the document's extracted gene list.
        year_filter:
            Exact year string matched against the document's extracted year.
        chunk_type_filter:
            ``"content"`` to retrieve only prose chunks,
            ``"metadata"`` to retrieve only structured metadata chunks,
            ``None`` (default) to retrieve both.
        allow_global_search:
            Overrides the instance default for this call only. When
            ``True``, a filter combination that matches nothing falls back
            to an unfiltered pass rather than returning empty.
        alpha:
            Override hybrid weight for this call only.

        Returns
        -------
        :class:`RetrievalResult` with ``context`` and ``chunks`` attributes.
        A ``RetrievalResult`` with empty ``context`` means nothing was found.
        """
        query = (query or "").strip()
        if not query:
            logger.warning("retrieve() called with empty query; returning empty result.")
            return RetrievalResult(context="", chunks=[])

        alpha = alpha if alpha is not None else self.alpha

        # -- Step 1: metadata filter -----------------------------------
        candidate_indices = self._apply_metadata_filters(
            source_filter=source_filter,
            section_filter=section_filter,
            disease_filter=disease_filter,
            gene_filter=gene_filter,
            year_filter=year_filter,
            chunk_type_filter=chunk_type_filter,
            paper_id_filter=paper_id_filter,
            document_id_filter=document_id_filter,
            allow_global_search=allow_global_search,
        )

        if not candidate_indices:
            logger.info("No candidates after metadata filtering for query: %r", query)
            return RetrievalResult(context="", chunks=[])

        # -- Step 2: dense retrieval over candidate pool -----------------
        query_vec = self.embedder.encode_query(query)          # shape (1, dim)

        # Build a temporary sub-index over filtered candidates.
        candidate_vectors = np.vstack(
            [self._faiss_vector_at(i) for i in candidate_indices]
        ).astype("float32")
        sub_index = faiss.IndexFlatIP(self.embedder.dimension)
        sub_index.add(candidate_vectors)

        n_search = min(len(candidate_indices), max(top_k * 3, 20))
        dense_scores_arr, local_indices = sub_index.search(query_vec, n_search)
        dense_scores_arr = dense_scores_arr[0]          # (n_search,)
        local_indices = local_indices[0]                 # (n_search,)

        # Map local sub-index positions back to global record indices.
        valid = local_indices >= 0
        global_indices = [candidate_indices[li] for li in local_indices[valid]]
        dense_scores = dense_scores_arr[valid].tolist()

        if not global_indices:
            return RetrievalResult(context="", chunks=[])

        # -- Step 3: keyword re-scoring -----------------------------------
        query_tfidf = self._tfidf.transform([normalize_for_embedding(query)])
        # Retrieve TF-IDF rows for global candidates.
        tfidf_rows = self._tfidf_matrix[global_indices]
        keyword_scores = (tfidf_rows @ query_tfidf.T).toarray().flatten().tolist()

        # -- Step 4: hybrid ranking -----------------------------------
        scored: list[tuple[float, int]] = []
        for dense, kw, gidx in zip(dense_scores, keyword_scores, global_indices):
            hybrid = alpha * float(dense) + (1.0 - alpha) * float(kw)
            if hybrid > _SCORE_FLOOR:
                scored.append((hybrid, gidx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        if not top:
            return RetrievalResult(context="", chunks=[])

        # -- Step 5: assemble context -----------------------------------
        result_chunks: list[dict] = []
        context_parts: list[str] = []

        for final_score, gidx in top:
            rec = dict(self._records[gidx])       # shallow copy
            rec["final_score"] = round(final_score, 4)
            result_chunks.append(rec)

            header = (
                f"[Source: {rec.get('source', '?')} | "
                f"Section: {rec.get('section', '?')} | "
                f"Type: {rec.get('chunk_type', 'content')} | "
                f"Page: {rec.get('page_number', '?')}]"
            )
            context_parts.append(f"{header}\n{rec['text']}")

        context = "\n\n".join(context_parts)

        logger.debug(
            "retrieve(): query=%r  top_k=%d  returned=%d",
            query, top_k, len(result_chunks),
        )
        return RetrievalResult(context=context, chunks=result_chunks)

    def retrieve_by_metadata(
        self,
        *,
        source_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
        disease_filter: Optional[str] = None,
        gene_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = "metadata",
        paper_id_filter: Optional[str] = None,
        document_id_filter: Optional[str] = None,
        allow_global_search: Optional[bool] = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Pure metadata retrieval -- no FAISS dense search involved.

        Useful for listing all papers that mention a specific disease or gene,
        or for surfacing structured metadata chunks directly.

        Parameters
        ----------
        All filter parameters behave identically to :meth:`retrieve`.
        chunk_type_filter defaults to ``"metadata"`` so callers get
        structured field chunks back, not prose passages.
        top_k:
            Maximum number of chunks returned.

        Returns
        -------
        :class:`RetrievalResult`.
        """
        indices = self._apply_metadata_filters(
            source_filter=source_filter,
            section_filter=section_filter,
            disease_filter=disease_filter,
            gene_filter=gene_filter,
            year_filter=year_filter,
            chunk_type_filter=chunk_type_filter,
            paper_id_filter=paper_id_filter,
            document_id_filter=document_id_filter,
            allow_global_search=allow_global_search,
        )

        result_chunks: list[dict] = []
        context_parts: list[str] = []

        for gidx in indices[:top_k]:
            rec = dict(self._records[gidx])
            rec["final_score"] = 1.0   # Metadata hits are considered exact.
            result_chunks.append(rec)

            header = (
                f"[Source: {rec.get('source', '?')} | "
                f"Section: {rec.get('section', '?')} | "
                f"Type: {rec.get('chunk_type', 'content')} | "
                f"Page: {rec.get('page_number', '?')}]"
            )
            context_parts.append(f"{header}\n{rec['text']}")

        context = "\n\n".join(context_parts)
        return RetrievalResult(context=context, chunks=result_chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _faiss_vector_at(self, global_index: int) -> np.ndarray:
        """Retrieve the stored FAISS vector for a given global record index."""
        vec = np.zeros((1, self.embedder.dimension), dtype="float32")
        self.index.reconstruct(global_index, vec[0])
        return vec

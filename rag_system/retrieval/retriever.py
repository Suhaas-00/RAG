"""Hybrid retriever: FAISS dense search + TF-IDF keyword re-scoring + metadata filtering.

The :class:`RAGRetriever` loads the serialised payload produced by
:mod:`rag_system.ingestion.DataIngestion` and exposes a single :meth:`retrieve`
method that:

1. **Metadata filter** (pre-retrieval)
   Narrows the candidate pool to chunks whose ``source`` or structured
   ``metadata`` fields match any supplied filters before FAISS search.
   Supported filters:

   - ``source_filter``    – exact or substring match on the PDF filename.
   - ``section_filter``   – exact match on the normalised section label.
   - ``disease_filter``   – substring match against the chunk's metadata
                            diseases list.
   - ``gene_filter``      – substring match against the chunk's metadata
                            genes list.
   - ``year_filter``      – exact match on the publication year string.
   - ``chunk_type_filter``– ``"content"``, ``"metadata"``, or ``None`` for both.

2. **Dense retrieval** (FAISS IndexFlatIP)
   Scores all filtered candidates by cosine similarity to the query vector.

3. **Keyword re-scoring** (TF-IDF dot product)
   Adds a weighted TF-IDF score to the dense score to lift exact-keyword
   matches that the embedding model may under-rank.

4. **Hybrid ranking**
   ``final_score = α × dense_score + (1-α) × keyword_score``
   where α is configurable (default 0.7).

5. **Context assembly**
   Top-k chunks are formatted as ``[Source: … | Section: … | Page: …]\n<text>``
   blocks joined by ``\n\n``.

Public API
----------
RetrievalResult(context, chunks)   – Named result container.
RAGRetriever.load(index_dir)       – Load from serialised payload.
RAGRetriever.retrieve(query, ...)  – Hybrid retrieval with metadata filtering.
RAGRetriever.retrieve_by_metadata(filters) – Pure metadata-based retrieval
                                            (no FAISS, no TF-IDF).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from rag_system.ingestion.embedding import PubMedEmbedder
from rag_system.utils.preprocessing import normalize_for_embedding

logger = logging.getLogger(__name__)

# Minimum schema version this retriever understands.
_MIN_SCHEMA_VERSION: int = 1
# Weight for dense score in hybrid ranking.
_DEFAULT_ALPHA: float = 0.7
# Score floor — chunks below this combined score are dropped.
_SCORE_FLOOR: float = 0.0


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

    Do not construct directly — use :meth:`load`.

    Parameters
    ----------
    index:      Loaded FAISS ``IndexFlatIP`` instance.
    payload:    Full metadata pickle dict produced by the ingestion pipeline.
    embedder:   :class:`~rag_system.ingestion.embedding.PubMedEmbedder` instance
                using the **same** model as ingestion.
    alpha:      Dense-score weight in hybrid ranking ``[0, 1]``.
                ``alpha=1`` → pure dense; ``alpha=0`` → pure keyword.
    """

    def __init__(
        self,
        index: faiss.Index,
        payload: dict,
        embedder: PubMedEmbedder,
        *,
        alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        self.index = index
        self.payload = payload
        self.embedder = embedder
        self.alpha = alpha

        self._records: list[dict] = payload["records"]
        self._id_to_pos: dict[str, int] = payload["id_to_position"]
        self._tfidf = payload["tfidf_vectorizer"]
        self._tfidf_matrix = payload["tfidf_matrix"]
        self._doc_metadata_index: dict[str, dict] = payload.get("doc_metadata_index", {})

        logger.info(
            "RAGRetriever ready — %d records, index.ntotal=%d",
            len(self._records),
            self.index.ntotal,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, index_dir: str | Path, *, alpha: float = _DEFAULT_ALPHA) -> "RAGRetriever":
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
        return cls(index, payload, embedder, alpha=alpha)

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
    ) -> list[int]:
        """Return FAISS row indices of records that pass all active filters.

        If *no* filter is supplied, all indices are returned (no narrowing).
        Filters are applied with AND logic (all must match).

        Matching rules
        --------------
        source_filter     – case-insensitive substring in ``record["source"]``.
        section_filter    – exact match on ``record["section"]`` after lowering.
        disease_filter    – case-insensitive substring in any disease in
                            ``record["metadata"]["diseases"]``.
        gene_filter       – case-insensitive substring in any gene in
                            ``record["metadata"]["genes"]``.
        year_filter       – exact match on ``record["metadata"]["year"]``.
        chunk_type_filter – exact match on ``record["chunk_type"]``.
        """
        has_filter = any(
            f is not None for f in (
                source_filter, section_filter, disease_filter,
                gene_filter, year_filter, chunk_type_filter,
            )
        )
        if not has_filter:
            return list(range(len(self._records)))

        src_lower = source_filter.lower() if source_filter else None
        sec_lower = section_filter.lower() if section_filter else None
        dis_lower = disease_filter.lower() if disease_filter else None
        gene_lower = gene_filter.lower() if gene_filter else None

        indices: list[int] = []
        for i, rec in enumerate(self._records):
            # ── source ──────────────────────────────────────────────────────
            if src_lower:
                rec_src = str(rec.get("source", "")).lower()
                if src_lower not in rec_src:
                    continue

            # ── section ─────────────────────────────────────────────────────
            if sec_lower:
                if str(rec.get("section", "")).lower() != sec_lower:
                    continue

            # ── chunk type ──────────────────────────────────────────────────
            if chunk_type_filter:
                if rec.get("chunk_type", "content") != chunk_type_filter:
                    continue

            # ── disease ─────────────────────────────────────────────────────
            if dis_lower:
                diseases = rec.get("metadata", {}).get("diseases", [])
                if not any(dis_lower in d.lower() for d in diseases):
                    continue

            # ── gene ─────────────────────────────────────────────────────────
            if gene_lower:
                genes = rec.get("metadata", {}).get("genes", [])
                if not any(gene_lower in g.lower() for g in genes):
                    continue

            # ── year ─────────────────────────────────────────────────────────
            if year_filter:
                rec_year = str(rec.get("metadata", {}).get("year", "") or "")
                if rec_year != str(year_filter):
                    continue

            indices.append(i)

        logger.debug(
            "_apply_metadata_filters: %d/%d records pass filters "
            "(source=%r section=%r disease=%r gene=%r year=%r type=%r)",
            len(indices), len(self._records),
            source_filter, section_filter, disease_filter,
            gene_filter, year_filter, chunk_type_filter,
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

        # ── Step 1: metadata filter ──────────────────────────────────────────
        candidate_indices = self._apply_metadata_filters(
            source_filter=source_filter,
            section_filter=section_filter,
            disease_filter=disease_filter,
            gene_filter=gene_filter,
            year_filter=year_filter,
            chunk_type_filter=chunk_type_filter,
        )

        if not candidate_indices:
            logger.info("No candidates after metadata filtering for query: %r", query)
            return RetrievalResult(context="", chunks=[])

        # ── Step 2: dense retrieval over candidate pool ──────────────────────
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

        # ── Step 3: keyword re-scoring ───────────────────────────────────────
        query_tfidf = self._tfidf.transform([normalize_for_embedding(query)])
        # Retrieve TF-IDF rows for global candidates.
        tfidf_rows = self._tfidf_matrix[global_indices]
        keyword_scores = (tfidf_rows @ query_tfidf.T).toarray().flatten().tolist()

        # ── Step 4: hybrid ranking ───────────────────────────────────────────
        scored: list[tuple[float, int]] = []
        for dense, kw, gidx in zip(dense_scores, keyword_scores, global_indices):
            hybrid = alpha * float(dense) + (1.0 - alpha) * float(kw)
            if hybrid > _SCORE_FLOOR:
                scored.append((hybrid, gidx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        if not top:
            return RetrievalResult(context="", chunks=[])

        # ── Step 5: assemble context ─────────────────────────────────────────
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
        top_k: int = 10,
    ) -> RetrievalResult:
        """Pure metadata retrieval — no FAISS dense search involved.

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
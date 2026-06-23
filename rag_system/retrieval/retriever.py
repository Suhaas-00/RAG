"""Production retrieval facade with validation, filtering, and context reconstruction.

Public API
----------
RAGRetriever.load(index_dir)          – Deserialise and validate a saved index.
RAGRetriever.retrieve(query, ...)     – Run hybrid search → rerank → build context.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss

from rag_system.ingestion.embedding import PubMedEmbedder
from rag_system.retrieval.hybrid_search import hybrid_search, is_noisy
from rag_system.retrieval.reranker import rerank
from rag_system.utils.config import Settings
from rag_system.utils.preprocessing import clean_text, token_count

logger = logging.getLogger(__name__)

# Bump this whenever the pickled payload schema changes.
_SUPPORTED_SCHEMA_VERSION: int = 1
_SUPPORTED_PREPROCESSING: str = "normalize_for_embedding:v1"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Immutable container returned by :meth:`RAGRetriever.retrieve`.

    Attributes
    ----------
    chunks:  Ranked list of metadata dicts (scores included).
    context: Pre-formatted, token-budgeted context string ready for an LLM.
    """

    chunks: list[dict] = field(default_factory=list)
    context: str = ""

    # Convenience predicate.
    @property
    def is_empty(self) -> bool:
        return not self.context


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RAGRetriever:
    """Thin facade over FAISS + PubMedEmbedder with context reconstruction.

    Do **not** construct directly — use :meth:`load`.
    """

    def __init__(
        self,
        index: faiss.Index,
        payload: dict,
        embedder: PubMedEmbedder,
        settings: Optional[Settings] = None,
    ) -> None:
        self.index = index
        self.payload = payload
        self.embedder = embedder
        self.settings: Settings = settings or Settings(model_name=payload["model_name"])
        # Build a fast chunk_id → record lookup table once at init.
        self.by_id: dict[str, dict] = {
            item["chunk_id"]: item for item in payload.get("records", [])
        }
        logger.info(
            "RAGRetriever ready — %d records, embedding_dim=%d, model=%s",
            len(self.by_id),
            index.d,
            payload["model_name"],
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        settings: Optional[Settings] = None,
    ) -> "RAGRetriever":
        """Deserialise a FAISS index + metadata pickle and validate consistency.

        Parameters
        ----------
        index_dir:
            Directory produced by the ingestion pipeline (contains
            ``vectors.index`` and ``metadata.pkl``).
        settings:
            Optional override for runtime settings. When provided, the
            ``model_name`` must match what is stored in the payload.

        Raises
        ------
        FileNotFoundError
            If either index file is missing.
        ValueError
            If schema version, preprocessing tag, model name, or embedding
            dimension do not match.
        """
        path = Path(index_dir)
        metadata_path = path / "metadata.pkl"
        vectors_path = path / "vectors.index"

        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.pkl not found in '{path}'")
        if not vectors_path.exists():
            raise FileNotFoundError(f"vectors.index not found in '{path}'")

        logger.info("Loading payload from %s", metadata_path)
        # pickle.load is intentionally restricted to indexes produced by this
        # trusted ingestion pipeline; do not load untrusted pickles.
        with metadata_path.open("rb") as fh:
            payload: dict = pickle.load(fh)  # nosec B301

        cls._validate_payload(payload, settings)

        model_name: str = payload["model_name"]
        expected_model: str = settings.model_name if settings else model_name

        logger.info("Loading FAISS index from %s", vectors_path)
        index: faiss.Index = faiss.read_index(str(vectors_path))

        embedder = PubMedEmbedder(expected_model)
        stored_dim: int = payload["embedding_dimension"]

        if index.d != stored_dim:
            raise ValueError(
                f"FAISS index dimension ({index.d}) ≠ payload dimension ({stored_dim}). "
                "Rebuild the index."
            )
        if index.d != embedder.dimension:
            raise ValueError(
                f"FAISS index dimension ({index.d}) ≠ embedder dimension ({embedder.dimension}). "
                "Rebuild the index or choose the correct model."
            )

        logger.info("Index loaded — ntotal=%d, dim=%d", index.ntotal, index.d)
        return cls(index, payload, embedder, settings)

    @staticmethod
    def _validate_payload(payload: dict, settings: Optional[Settings]) -> None:
        """Raise ``ValueError`` for any schema or configuration mismatch."""
        schema_ver = payload.get("schema_version")
        if schema_ver != _SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version '{schema_ver}'. "
                f"Expected {_SUPPORTED_SCHEMA_VERSION}. Rebuild the index."
            )
        preprocessing = payload.get("preprocessing")
        if preprocessing != _SUPPORTED_PREPROCESSING:
            raise ValueError(
                f"Unsupported preprocessing '{preprocessing}'. "
                f"Expected '{_SUPPORTED_PREPROCESSING}'. Rebuild the index."
            )
        if settings and settings.model_name != payload.get("model_name"):
            raise ValueError(
                f"Embedding model mismatch: "
                f"index='{payload.get('model_name')}', "
                f"settings='{settings.model_name}'."
            )

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(
        self,
        anchors: list[dict],
        section_filter: Optional[str] = None,
    ) -> str:
        """Reconstruct a token-budgeted context string from ranked anchor chunks.

        For each anchor the preceding and following chunks (by ``prev_chunk_id``
        / ``next_chunk_id``) are included when they belong to the same source,
        are not noisy, and fit within :attr:`Settings.max_context_tokens`.

        Parameters
        ----------
        anchors:        Ranked list of chunk dicts (output of :func:`rerank`).
        section_filter: When set, only chunks whose ``section`` field matches
                        (case-insensitive) are included.

        Returns
        -------
        A multi-block string with provenance headers, ready for an LLM prompt.
        """
        blocks: list[str] = []
        used_ids: set[str] = set()
        used_tokens: int = 0
        max_tokens: int = self.settings.max_context_tokens

        for anchor in anchors:
            if used_tokens >= max_tokens:
                break

            # Build the ordered window: [prev, anchor, next].
            window_ids = [
                anchor.get("prev_chunk_id"),
                anchor.get("chunk_id"),
                anchor.get("next_chunk_id"),
            ]
            parts: list[str] = []

            for chunk_id in window_ids:
                if chunk_id is None:
                    continue
                item = self.by_id.get(chunk_id)
                if item is None:
                    continue
                if item["chunk_id"] in used_ids:
                    continue
                # Same-source constraint.
                if item.get("source") != anchor.get("source"):
                    continue
                # Optional section constraint.
                if section_filter and item.get("section", "").casefold() != section_filter.casefold():
                    continue
                # Quality gate.
                if is_noisy(item.get("text", "")):
                    continue

                text = self._repair_text(item["text"])
                cost = token_count(text)
                remaining = max_tokens - used_tokens

                if remaining <= 0:
                    break
                if cost > remaining:
                    # Truncate to the token budget at the word boundary.
                    words = text.split()
                    text = " ".join(words[:remaining])
                    cost = token_count(text)

                parts.append(text)
                used_ids.add(item["chunk_id"])
                used_tokens += cost

            if parts:
                header = (
                    f"[Source: {anchor.get('source', 'unknown')} | "
                    f"Section: {anchor.get('section', 'unknown')} | "
                    f"Page: {anchor.get('page_number', '?')}]"
                )
                blocks.append(header + "\n" + " ".join(parts))

        return "\n\n".join(blocks)

    @staticmethod
    def _repair_text(text: str) -> str:
        """Apply lightweight repairs to already-indexed chunks."""
        text = clean_text(text)
        # Soft-hyphen word joins that may survive clean_text.
        text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
        # Known run-together compound from PubMedBERT tokenisation.
        text = re.sub(r"\boncogeneaddicted\b", "oncogene-addicted", text, flags=re.IGNORECASE)
        return text

    # ------------------------------------------------------------------
    # Public retrieval method
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        source_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Run hybrid search → rerank → context reconstruction.

        Parameters
        ----------
        query:          Natural-language or cleaned query string.
        source_filter:  Optional PDF filename (``"12345678.pdf"``) to constrain
                        retrieval to a single document.
        section_filter: Optional section name (``"abstract"``, ``"methods"``, …)
                        to constrain retrieval to one section.
        top_k:          Override :attr:`Settings.top_k` for this call.

        Returns
        -------
        A :class:`RetrievalResult` with ``chunks`` and ``context`` populated,
        or an empty result when the query is blank or no candidates pass the
        quality threshold.
        """
        query = clean_text(query, lowercase=True)
        if not query or not any(ch.isalnum() for ch in query):
            logger.debug("Empty or non-alphanumeric query after cleaning; returning empty result.")
            return RetrievalResult()

        effective_top_k = top_k if top_k is not None else self.settings.top_k

        logger.debug(
            "retrieve(query=%r, source=%r, section=%r, top_k=%d)",
            query,
            source_filter,
            section_filter,
            effective_top_k,
        )

        candidates = hybrid_search(
            query=query,
            embedder=self.embedder,
            index=self.index,
            payload=self.payload,
            candidate_k=self.settings.candidate_k,
            semantic_weight=self.settings.semantic_weight,
            keyword_weight=self.settings.keyword_weight,
            source_filter=source_filter,
            section_filter=section_filter,
        )

        ranked = rerank(query, candidates, top_k=effective_top_k)

        # Suppress results with no meaningful semantic or lexical signal.
        # (Section-filtered queries are exempt because they are already narrow.)
        if not section_filter:
            before = len(ranked)
            ranked = [
                item for item in ranked
                if (
                    item.get("semantic_score_raw", 0.0) >= 0.20
                    or item.get("keyword_score_raw", 0.0) >= 0.05
                )
            ]
            if len(ranked) < before:
                logger.debug(
                    "Quality filter removed %d low-confidence chunk(s).",
                    before - len(ranked),
                )

        context = self._build_context(ranked, section_filter)

        logger.debug(
            "retrieve() → %d ranked chunk(s), context_tokens≈%d",
            len(ranked),
            token_count(context),
        )

        return RetrievalResult(chunks=ranked, context=context)
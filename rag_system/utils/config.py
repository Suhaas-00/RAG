"""Single source of truth for ingestion and query-time settings.

All runtime-tunable parameters live here so that the rest of the codebase
imports :class:`Settings` rather than scattering magic constants.

Usage
-----
::

    from rag_system.utils.config import Settings

    cfg = Settings()                           # all defaults
    cfg = Settings(chunk_size=512)             # override one field
    cfg = Settings.from_env()                  # read overrides from env vars
    cfg = Settings.from_dict({"top_k": 5})    # override from a plain dict

Environment variables (optional overrides)
-----------------------------------------
``RAG_MODEL_NAME``          – HuggingFace embedding model identifier.
``RAG_PDF_DIR``             – Directory containing source PDFs.
``RAG_OUTPUT_DIR``          – Root output directory (index lives at outputs/index/faiss_index).
``RAG_CHUNK_SIZE``          – Approximate token size of each chunk.
``RAG_CHUNK_OVERLAP``       – Token overlap between consecutive chunks.
``RAG_EMBEDDING_BATCH``     – Number of texts to encode per batch.
``RAG_SEMANTIC_WEIGHT``     – Hybrid search semantic weight (0.0–1.0).
``RAG_KEYWORD_WEIGHT``      – Hybrid search keyword weight (0.0–1.0).
``RAG_CANDIDATE_K``         – FAISS nearest-neighbour candidates to retrieve.
``RAG_TOP_K``               – Final number of chunks returned to the caller.
``RAG_MAX_CONTEXT_TOKENS``  – Token budget for the LLM context window.
``RAG_MIN_CHUNK_TOKENS``    – Minimum tokens required to keep a chunk.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_PREFIX = "RAG_"


def _env(name: str, default: str) -> str:
    """Return the environment variable ``RAG_<NAME>`` or *default*."""
    return os.environ.get(f"{_ENV_PREFIX}{name}", default)


def _existing_or_default(primary: str, legacy: str) -> Path:
    """Prefer the organized path, but keep older checkouts runnable."""
    primary_path = Path(primary)
    legacy_path = Path(legacy)
    if primary_path.exists() or not legacy_path.exists():
        return primary_path
    return legacy_path


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Immutable configuration container for the RAG pipeline.

    Attributes
    ----------
    model_name:
        HuggingFace model identifier used for all dense embeddings.
        **Must** match the model used to build the FAISS index.
    pdf_dir:
        Directory scanned for source PDF files during ingestion.
    output_dir:
        Root directory for pipeline artefacts; the FAISS index is written
        to ``output_dir / "faiss_index"``.
    chunk_size:
        Target chunk size in tokens (approximate; splitter may vary slightly).
    chunk_overlap:
        Token overlap between consecutive chunks to preserve context across
        boundaries.
    embedding_batch_size:
        Number of text chunks encoded per :meth:`PubMedEmbedder.encode` call.
        Larger values use more memory but are faster on GPU.
    semantic_weight:
        Weight applied to the FAISS inner-product score during hybrid ranking.
        Must satisfy ``semantic_weight + keyword_weight ≤ 1.0``
        (section boosts may push the total above 1.0).
    keyword_weight:
        Weight applied to the keyword overlap score during hybrid ranking.
    candidate_k:
        Number of FAISS nearest neighbours retrieved before reranking.
    top_k:
        Number of chunks returned to the caller after reranking.
    max_context_tokens:
        Hard token budget for the context string assembled from retrieved chunks.
    min_chunk_tokens:
        Chunks with fewer than this many tokens are discarded during ingestion.
    """

    model_name: str = "NeuML/pubmedbert-base-embeddings"
    pdf_dir: Path = field(default_factory=lambda: _existing_or_default("datasets/raw/pdfs", "pdfs"))
    output_dir: Path = field(default_factory=lambda: _existing_or_default("outputs/index", "output"))
    chunk_size: int = 400
    chunk_overlap: int = 64
    embedding_batch_size: int = 32
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    candidate_k: int = 10
    top_k: int = 3
    max_context_tokens: int = 2000
    min_chunk_tokens: int = 40

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def index_dir(self) -> Path:
        """Absolute path to the FAISS index directory."""
        return self.output_dir / "faiss_index"

    @property
    def metadata_path(self) -> Path:
        """Absolute path to the serialised metadata pickle."""
        return self.index_dir / "metadata.pkl"

    @property
    def vectors_path(self) -> Path:
        """Absolute path to the serialised FAISS vectors."""
        return self.index_dir / "vectors.index"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raise :exc:`ValueError` if any setting is out of its valid range."""
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be ≥ 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if not (0.0 <= self.semantic_weight <= 1.0):
            raise ValueError(
                f"semantic_weight must be in [0.0, 1.0], got {self.semantic_weight}"
            )
        if not (0.0 <= self.keyword_weight <= 1.0):
            raise ValueError(
                f"keyword_weight must be in [0.0, 1.0], got {self.keyword_weight}"
            )
        if self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be ≥ top_k ({self.top_k})"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be ≥ 1, got {self.top_k}")
        if self.max_context_tokens < 1:
            raise ValueError(
                f"max_context_tokens must be ≥ 1, got {self.max_context_tokens}"
            )
        if self.min_chunk_tokens < 1:
            raise ValueError(
                f"min_chunk_tokens must be ≥ 1, got {self.min_chunk_tokens}"
            )
        if self.embedding_batch_size < 1:
            raise ValueError(
                f"embedding_batch_size must be ≥ 1, got {self.embedding_batch_size}"
            )

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Settings":
        """Construct a :class:`Settings` instance from environment variables.

        Reads ``RAG_*`` environment variables; falls back to the class
        defaults for any variable that is not set.

        Example
        -------
        ::

            $ RAG_CHUNK_SIZE=512 RAG_TOP_K=5 python my_script.py

        """
        return cls(
            model_name=_env("MODEL_NAME", "NeuML/pubmedbert-base-embeddings"),
            pdf_dir=Path(_env("PDF_DIR", str(_existing_or_default("datasets/raw/pdfs", "pdfs")))),
            output_dir=Path(_env("OUTPUT_DIR", str(_existing_or_default("outputs/index", "output")))),
            chunk_size=int(_env("CHUNK_SIZE", "400")),
            chunk_overlap=int(_env("CHUNK_OVERLAP", "64")),
            embedding_batch_size=int(_env("EMBEDDING_BATCH", "32")),
            semantic_weight=float(_env("SEMANTIC_WEIGHT", "0.7")),
            keyword_weight=float(_env("KEYWORD_WEIGHT", "0.3")),
            candidate_k=int(_env("CANDIDATE_K", "10")),
            top_k=int(_env("TOP_K", "3")),
            max_context_tokens=int(_env("MAX_CONTEXT_TOKENS", "2000")),
            min_chunk_tokens=int(_env("MIN_CHUNK_TOKENS", "40")),
        )

    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> "Settings":
        """Create a :class:`Settings` instance from a plain dict.

        Only keys that correspond to known fields are applied; unknown keys
        are logged and ignored.

        Parameters
        ----------
        overrides:
            Mapping of field names → values.  ``Path``-typed fields accept
            both ``str`` and :class:`pathlib.Path` values.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            logger.warning("Settings.from_dict: unknown keys ignored: %s", sorted(unknown))

        kwargs: dict[str, Any] = {}
        defaults = cls()  # Use defaults as the base.
        for f in fields(cls):
            if f.name in overrides:
                value = overrides[f.name]
                # Coerce str → Path for Path-typed fields.
                if f.type in (Path, "Path") and isinstance(value, str):
                    value = Path(value)
                kwargs[f.name] = value
            else:
                kwargs[f.name] = getattr(defaults, f.name)

        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable ``dict`` representation of this instance."""
        return {
            f.name: str(getattr(self, f.name)) if isinstance(getattr(self, f.name), Path)
            else getattr(self, f.name)
            for f in fields(self)
        }

    def __repr__(self) -> str:
        pairs = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self))
        return f"Settings({pairs})"

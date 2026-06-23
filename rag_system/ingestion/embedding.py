"""One embedding implementation shared by ingestion and retrieval.

The :class:`PubMedEmbedder` wraps a ``sentence-transformers`` model and
enforces the normalisation contract required for FAISS ``IndexFlatIP``
(inner-product == cosine similarity when vectors are unit-normalised).

Design principles
-----------------
* **Shared normalisation** — all text passes through
  :func:`~rag_system.utils.preprocessing.normalize_for_embedding` before
  encoding so that ingestion-time and query-time representations are identical.
* **Defensive re-normalisation** — even though ``SentenceTransformer`` can
  normalise internally, we apply a NumPy L2-norm pass afterwards.  This
  guards against floating-point drift and ensures FAISS scores stay in
  ``[0, 1]``.
* **Progress bar gate** — shown only when the batch list is larger than a
  single batch, avoiding noise in unit tests or single-query paths.
* **Device auto-detection** — when *device* is ``None`` the model picks the
  fastest available backend (CUDA → MPS → CPU).

Public API
----------
PubMedEmbedder(model_name, device)   – Load a SentenceTransformer model.
PubMedEmbedder.dimension             – Embedding vector size (int property).
PubMedEmbedder.encode(texts, ...)    – Encode and L2-normalise a text list.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from rag_system.utils.preprocessing import normalize_for_embedding

logger = logging.getLogger(__name__)

# Minimum float norm used in the safe-division guard.
_NORM_EPS: float = 1e-12


class PubMedEmbedder:
    """Dense-vector encoder backed by a ``sentence-transformers`` model.

    Parameters
    ----------
    model_name:
        A HuggingFace model identifier or local path understood by
        ``SentenceTransformer``.  The default used throughout this project
        is ``"NeuML/pubmedbert-base-embeddings"`` (768-dimensional).
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``).
        Pass ``None`` to let ``sentence-transformers`` auto-detect the
        fastest available device.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.
    """

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for PubMedEmbedder. "
                "Install it with:  pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        logger.info("Loading embedding model '%s' (device=%s)", model_name, device or "auto")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(
            "Embedding model ready — dim=%d, device=%s",
            self.dimension,
            getattr(self.model, "device", "unknown"),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality.

        Prefers the current ``get_embedding_dimension()`` API while retaining
        backward compatibility with older ``sentence-transformers`` releases
        that expose only ``get_sentence_embedding_dimension()``.
        """
        getter = getattr(self.model, "get_embedding_dimension", None)
        if callable(getter):
            return int(getter())
        return int(self.model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        show_progress_bar: Optional[bool] = None,
    ) -> np.ndarray:
        """Encode *texts* and return unit-normalised float32 vectors.

        Parameters
        ----------
        texts:
            List of raw text strings.  Each is normalised via
            :func:`~rag_system.utils.preprocessing.normalize_for_embedding`
            before being fed to the model.
        batch_size:
            Number of texts encoded per forward pass.  Larger values reduce
            wall-clock time on GPU but increase memory consumption.
        show_progress_bar:
            When ``None`` (default), a progress bar is shown only when
            ``len(texts) > batch_size``.  Pass ``True`` or ``False`` to
            override.

        Returns
        -------
        ``np.ndarray`` of shape ``(len(texts), dimension)`` and dtype
        ``float32``.  Every row is unit-normalised so that inner product ==
        cosine similarity.

        Raises
        ------
        ValueError
            If *texts* is empty or *batch_size* is < 1.
        """
        if not texts:
            raise ValueError("encode() received an empty text list.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {batch_size}")

        # Apply the shared normalisation contract.
        prepared = [normalize_for_embedding(t) for t in texts]

        # Auto-decide progress bar visibility.
        if show_progress_bar is None:
            show_progress_bar = len(prepared) > batch_size

        logger.debug(
            "Encoding %d text(s) with batch_size=%d, progress_bar=%s",
            len(prepared),
            batch_size,
            show_progress_bar,
        )

        vectors = self.model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,   # SentenceTransformer-level normalisation.
        )

        vectors = np.asarray(vectors, dtype="float32")

        # Defensive L2 re-normalisation (guards against floating-point drift).
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, _NORM_EPS)

        logger.debug("encode() → shape=%s, dtype=%s", vectors.shape, vectors.dtype)
        return vectors

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"PubMedEmbedder(model_name={self.model_name!r}, dimension={self.dimension})"
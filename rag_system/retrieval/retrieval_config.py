"""Project-wide retrieval scope configuration.

The default retrieval mode is global. Document identifiers such as
``paper_id`` or ``source`` are only applied when document scope is enabled
through this module or by an explicit per-call override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RetrievalConfig:
    """Configuration for retrieval document-scope behavior."""

    enable_document_scope: bool = _env_bool("RAG_ENABLE_DOCUMENT_SCOPE", False)


config = RetrievalConfig()

# Backward-friendly module constant for simple programmatic use:
#     from rag_system.retrieval import retrieval_config
#     retrieval_config.ENABLE_DOCUMENT_SCOPE = True
ENABLE_DOCUMENT_SCOPE: bool = config.enable_document_scope


def document_scope_enabled(override: bool | None = None) -> bool:
    """Return the effective document-scope setting."""
    if override is not None:
        return override
    return bool(ENABLE_DOCUMENT_SCOPE)


def set_document_scope(enabled: bool) -> None:
    """Set document scope at process runtime."""
    global ENABLE_DOCUMENT_SCOPE
    config.enable_document_scope = enabled
    ENABLE_DOCUMENT_SCOPE = enabled

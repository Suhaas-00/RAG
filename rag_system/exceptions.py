"""Typed exceptions for the RAG platform."""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for expected platform failures."""


class ConfigurationError(RAGError):
    """Raised when runtime configuration is invalid."""


class IndexUnavailableError(RAGError):
    """Raised when a required retrieval index artifact is missing or invalid."""


class RetrievalError(RAGError):
    """Raised when retrieval cannot complete successfully."""


class GenerationError(RAGError):
    """Raised when answer generation fails."""


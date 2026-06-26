"""Compatibility wrapper for the top-level DataIngestion module."""

from DataIngestion import ingest, load_pdf, main

__all__ = ["ingest", "load_pdf", "main"]

"""Backward-compatible CLI shim for the production RAG ingestion package.

This module exists solely to preserve the original entry-point contract.
All logic lives in :mod:`rag_system.ingestion.DataIngestion`.

Usage
-----
::

    python -m rag_system.ingestion.ingest_cli [--pdf-dir PATH] [--output-dir PATH]

or, if registered as a console script in ``pyproject.toml``::

    rag-ingest [--pdf-dir PATH] [--output-dir PATH]
"""

from rag_system.ingestion.DataIngestion import main

if __name__ == "__main__":
    main()
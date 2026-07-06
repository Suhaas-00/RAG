"""Document-scope metadata filtering shared by all retrieval paths.

Behavior summary
-----------------
* Document-scope filtering is controlled by
  :mod:`rag_system.retrieval.retrieval_config` and is disabled by default.
* When document scope is enabled and a caller supplies a document identifier
  (``source``, ``paper_id``, ``document_id``, etc.), retrieval is scoped
  strictly to matching document(s) -- other documents are excluded even if
  they'd otherwise be relevant.
* If no document identifier is supplied, retrieval defaults to searching
  the **entire corpus** rather than returning nothing. A user who hasn't
  named a specific paper should still get an answer.
* This can be overridden in either direction via ``allow_global_search`` /
  ``require_document_scope`` on :func:`filter_record_indices` -- see that
  function's docstring for the exact decision table.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_system.retrieval.retrieval_config import document_scope_enabled

logger = logging.getLogger(__name__)

DOCUMENT_ID_KEYS: tuple[str, ...] = ("paper_id", "document_id", "doc_id", "source_id", "file_id")
SOURCE_KEYS: tuple[str, ...] = ("source", "filename", "file_name")
DOCUMENT_FILTER_KEYS: tuple[str, ...] = DOCUMENT_ID_KEYS + SOURCE_KEYS


def _as_values(value: Any) -> list[str]:
    """Normalize a filter value (scalar, string, or iterable) to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _record_value(record: dict[str, Any], key: str) -> str | None:
    """Look up *key* on a record, falling back to a nested ``metadata`` dict."""
    value = record.get(key)
    if value is None and isinstance(record.get("metadata"), dict):
        value = record["metadata"].get(key)
    if value is None:
        return None
    return str(value)


def _same_identifier(actual: str | None, wanted: str) -> bool:
    """Case-insensitive identifier match, tolerant of path vs. filename vs. stem."""
    if not actual:
        return False
    actual_fold = actual.casefold()
    wanted_fold = wanted.casefold()
    if actual_fold == wanted_fold:
        return True
    return (
        Path(actual_fold).name == Path(wanted_fold).name
        or Path(actual_fold).stem == Path(wanted_fold).stem
    )


def document_filter_values(filters: dict[str, Any]) -> dict[str, list[str]]:
    """Return normalized document-scope filter values from a filter mapping."""
    values: dict[str, list[str]] = {}
    aliases = {
        "paper_ids": "paper_id",
        "document_ids": "document_id",
        "sources": "source",
        "source_filter": "source",
        "source_filters": "source",
        "paper_id_filter": "paper_id",
        "document_id_filter": "document_id",
    }
    for key, raw in filters.items():
        canonical = aliases.get(key, key)
        if canonical in DOCUMENT_FILTER_KEYS:
            values.setdefault(canonical, []).extend(_as_values(raw))
    return {key: val for key, val in values.items() if val}


def has_document_scope(filters: dict[str, Any]) -> bool:
    """Return True when filters contain at least one document identifier."""
    return bool(document_filter_values(filters))


def document_matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True when *record* matches any requested document identifier group.

    Different document identifier keys are treated as aliases, not as
    independent AND constraints. This allows callers to provide both
    ``paper_id`` and ``source`` for the same document without excluding
    legacy records that only carry one of those fields.
    """
    wanted = document_filter_values(filters)
    if not wanted:
        return True

    for filter_key, values in wanted.items():
        candidate_keys = SOURCE_KEYS if filter_key in SOURCE_KEYS else DOCUMENT_ID_KEYS
        for value in values:
            if any(_same_identifier(_record_value(record, key), value) for key in candidate_keys):
                return True
    return False


def metadata_matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Apply non-document metadata filters (section, chunk type, disease, gene, year)."""
    section = filters.get("section") or filters.get("section_filter")
    if section and str(record.get("section", "")).casefold() != str(section).casefold():
        return False

    chunk_type = filters.get("chunk_type") or filters.get("chunk_type_filter")
    if chunk_type and str(record.get("chunk_type", "content")) != str(chunk_type):
        return False

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}

    disease = filters.get("disease") or filters.get("disease_filter")
    if disease:
        diseases = metadata.get("diseases", [])
        if not any(str(disease).casefold() in str(item).casefold() for item in diseases):
            return False

    gene = filters.get("gene") or filters.get("gene_filter")
    if gene:
        genes = metadata.get("genes", [])
        if not any(str(gene).casefold() in str(item).casefold() for item in genes):
            return False

    year = filters.get("year") or filters.get("year_filter")
    if year and str(metadata.get("year", "") or "") != str(year):
        return False

    return True


def filter_record_indices(
    records: list[dict[str, Any]],
    filters: dict[str, Any] | None = None,
    *,
    enable_document_filtering: bool | None = None,
    allow_global_search: bool = True,
    require_document_scope: bool = True,
    drop_noisy: bool = False,
    noisy_predicate: Any | None = None,
) -> list[int]:
    """Return indices of records satisfying document and metadata filters.

    Decision table
    ---------------
    1. ``enable_document_filtering=False``
       -> document-identity filtering is skipped entirely; metadata filters
          like section/year/gene/chunk_type still apply.

    2. A document identifier is supplied (``source`` / ``paper_id`` /
       ``document_id`` / ...)
       -> strictly scoped: only records matching that document pass,
          regardless of ``allow_global_search``. This never silently
          expands to the whole corpus, even if the identifier matches
          nothing -- callers that want a fallback should retry explicitly
          without the identifier.

    3. No document identifier is supplied
       -> ``allow_global_search=True`` (the default): search the entire
          corpus. This is the common case for open-ended questions where
          the user hasn't named a specific paper.
       -> ``allow_global_search=False`` *and* ``require_document_scope=True``:
          return ``[]`` and log a debug message. Use this combination only when
          you deliberately want to refuse to answer without an explicit
          document scope.
       -> ``allow_global_search=False`` and ``require_document_scope=False``:
          search the entire corpus anyway (kept for backward compatibility
          with callers that manage the guard themselves).

    Metadata filters (section/year/gene/etc.) are always applied on top of
    whatever document-identity decision is made above.
    """
    filters = filters or {}
    scoped = has_document_scope(filters)
    effective_enabled = document_scope_enabled(enable_document_filtering)

    # ------------------------------------------------------------
    # Decide whether document-identity filtering should be applied
    # ------------------------------------------------------------
    if not effective_enabled:
        apply_document_filter = False
    elif scoped:
        logger.debug("Document-scoped retrieval selected for filters: %s", document_filter_values(filters))
        apply_document_filter = True
    elif allow_global_search:
        logger.debug("Global retrieval selected (no document scope).")
        apply_document_filter = False
    elif require_document_scope:
        logger.debug(
            "Document-scope filtering is enabled but no document identifier "
            "was supplied, and allow_global_search=False was set explicitly; "
            "returning no candidates."
        )
        return []
    else:
        apply_document_filter = False

    # ------------------------------------------------------------
    # Filter records
    # ------------------------------------------------------------
    indices: list[int] = []
    for idx, record in enumerate(records):
        if drop_noisy and noisy_predicate is not None and noisy_predicate(record.get("text", "")):
            continue
        if apply_document_filter and not document_matches(record, filters):
            continue
        if not metadata_matches(record, filters):
            continue
        indices.append(idx)

    if apply_document_filter and not indices:
        logger.info("No records matched document filters: %s", document_filter_values(filters))

    logger.debug(
        "Document scope: enabled=%s, scoped=%s, apply_document_filter=%s, allow_global_search=%s, returned=%d",
        effective_enabled, scoped, apply_document_filter, allow_global_search, len(indices),
    )
    return indices

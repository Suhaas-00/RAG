"""Document-scope metadata filtering shared by all retrieval paths."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DOCUMENT_ID_KEYS: tuple[str, ...] = ("paper_id", "document_id", "doc_id", "source_id", "file_id")
SOURCE_KEYS: tuple[str, ...] = ("source", "filename", "file_name")
DOCUMENT_FILTER_KEYS: tuple[str, ...] = DOCUMENT_ID_KEYS + SOURCE_KEYS


def _as_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _record_value(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value is None and isinstance(record.get("metadata"), dict):
        value = record["metadata"].get(key)
    if value is None:
        return None
    return str(value)


def _same_identifier(actual: str | None, wanted: str) -> bool:
    if not actual:
        return False
    actual_fold = actual.casefold()
    wanted_fold = wanted.casefold()
    if actual_fold == wanted_fold:
        return True
    return Path(actual_fold).name == Path(wanted_fold).name or Path(actual_fold).stem == Path(wanted_fold).stem


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
    """Apply non-document metadata filters after document scope is checked."""
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
    enable_document_filtering: bool = True,
    allow_global_search: bool = False,
    require_document_scope: bool = True,
    drop_noisy: bool = False,
    noisy_predicate: Any | None = None,
) -> list[int]:
    """Return record indices that satisfy document and metadata filters."""
    filters = filters or {}
    scoped = has_document_scope(filters)

    if enable_document_filtering and require_document_scope and not scoped and not allow_global_search:
        logger.warning(
            "Document-scope filtering is enabled but no document identifier was supplied; "
            "returning no candidates. Pass allow_global_search=True for explicit global retrieval."
        )
        return []

    indices: list[int] = []
    for idx, record in enumerate(records):
        if drop_noisy and noisy_predicate is not None and noisy_predicate(record.get("text", "")):
            continue
        if enable_document_filtering and scoped and not document_matches(record, filters):
            continue
        if not metadata_matches(record, filters):
            continue
        indices.append(idx)

    if enable_document_filtering and scoped and not indices:
        logger.info("No records matched document-scope filters: %s", document_filter_values(filters))
    return indices

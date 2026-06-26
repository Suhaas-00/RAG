"""Intent parser for medical RAG queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rag_system.cleaner import CANONICAL_SECTIONS, canonical_section


@dataclass(frozen=True)
class QueryIntent:
    intent: str
    query: str
    paper_label: str | None = None
    paper_source: str | None = None
    section: str | None = None
    disease: str | None = None
    metadata_field: str | None = None
    filters: dict[str, str] = field(default_factory=dict)


PAPER_REF_RE = re.compile(r"\bpaper\s*(\d+)\b", re.IGNORECASE)
PDF_RE = re.compile(r"\b([\w.-]+\.pdf)\b", re.IGNORECASE)
PMID_RE = re.compile(r"\b(\d{6,10})\b")


def _detect_section(query: str) -> str | None:
    lowered = query.lower()
    aliases = sorted(CANONICAL_SECTIONS, key=len, reverse=True)
    for alias in aliases:
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return canonical_section(alias)
    return None


def _resolve_paper(query: str, paper_map: dict[str, str]) -> tuple[str | None, str | None]:
    match = PAPER_REF_RE.search(query)
    if match:
        label = f"paper {int(match.group(1))}"
        return label, paper_map.get(label)

    pdf = PDF_RE.search(query)
    if pdf:
        wanted = pdf.group(1)
        for label, filename in paper_map.items():
            if Path(filename).name.casefold() == wanted.casefold():
                return label, filename
        return None, wanted

    pmid = PMID_RE.search(query)
    if pmid:
        token = pmid.group(1)
        for label, filename in paper_map.items():
            if Path(filename).stem == token:
                return label, filename
    return None, None


def parse_query(query: str, paper_map: dict[str, str] | None = None) -> QueryIntent:
    paper_map = paper_map or {}
    cleaned = (query or "").strip()
    lowered = cleaned.lower()
    paper_label, paper_source = _resolve_paper(cleaned, paper_map)
    section = _detect_section(cleaned)

    filters: dict[str, str] = {}
    if paper_source:
        filters["source"] = paper_source
    if paper_label:
        filters["paper_id"] = paper_label
    if section:
        filters["section"] = section

    if re.search(r"\b(list|show)\b.*\b(papers?|pdfs?|documents?)\b", lowered):
        return QueryIntent("list_papers", cleaned)

    for field in ("doi", "pmid", "year", "journal", "authors", "title"):
        if re.search(rf"\b{field}\b", lowered):
            return QueryIntent("metadata_query", cleaned, paper_label, paper_source, section, metadata_field=field, filters=filters)

    if paper_source and section:
        return QueryIntent("section_lookup", cleaned, paper_label, paper_source, section, filters=filters)

    if paper_source:
        return QueryIntent("paper_lookup", cleaned, paper_label, paper_source, section, filters=filters)

    disease_match = re.search(
        r"\b(cancer|carcinoma|diabetes|hypertension|asthma|tuberculosis|pneumonia|sepsis|stroke|disease|condition)\b",
        lowered,
    )
    if disease_match:
        disease = None if disease_match.group(1) in {"disease", "condition"} else disease_match.group(1)
        return QueryIntent("disease_lookup", cleaned, paper_label, paper_source, section, disease=disease, filters=filters)

    if section:
        return QueryIntent("section_lookup", cleaned, paper_label, paper_source, section, filters=filters)

    return QueryIntent("semantic_qa", cleaned, paper_label, paper_source, section, filters=filters)


def build_retrieval_query(intent: QueryIntent) -> str:
    query = intent.query.lower()
    query = PAPER_REF_RE.sub(" ", query)
    query = PDF_RE.sub(" ", query)
    query = re.sub(r"\b(?:give|show|tell|find|what|which|is|are|the|of|in|for|about|paper|section)\b", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    if intent.section and intent.section not in query:
        query = f"{intent.section} {query}".strip()
    return query or intent.query


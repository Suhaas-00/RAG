"""Biomedical query understanding for document-first retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag_system.retrieval.identifier_patterns import (
    DEFAULT_IDENTIFIER_PATTERNS,
    IdentifierPattern,
    normalize_identifier,
)


class QueryIntentType(str, Enum):
    SAMPLE_SIZE = "sample_size"
    CONCLUSIONS = "conclusions"
    METHODOLOGY = "methodology"
    MUTATION = "mutation"
    ENTITY = "entity"
    SAFETY = "safety"
    EFFICACY = "efficacy"
    METADATA = "metadata"
    GENERIC = "generic"


@dataclass(frozen=True)
class DetectedIdentifier:
    type: str
    value: str
    normalized: str
    span: tuple[int, int]


@dataclass(frozen=True)
class QueryUnderstanding:
    original_query: str
    expanded_query: str
    intent: QueryIntentType
    identifiers: tuple[DetectedIdentifier, ...] = ()
    entities: tuple[str, ...] = ()
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


_INTENT_RULES: tuple[tuple[QueryIntentType, tuple[str, ...]], ...] = (
    (QueryIntentType.SAMPLE_SIZE, ("sample size", "how many patients", "screened", "randomized", "randomised", "enrolled", "assigned")),
    (QueryIntentType.CONCLUSIONS, ("conclude", "concluded", "conclusion", "authors conclude", "interpretation")),
    (QueryIntentType.METHODOLOGY, ("method", "methods", "study design", "trial design", "protocol", "randomization")),
    (QueryIntentType.MUTATION, ("mutation", "variant", "exon", "amino acid", "pathway")),
    (QueryIntentType.SAFETY, ("adverse event", "toxicity", "safety", "serious adverse")),
    (QueryIntentType.EFFICACY, ("response rate", "orr", "pfs", "overall survival", "efficacy", "hazard ratio")),
    (QueryIntentType.METADATA, ("doi", "pmid", "pmcid", "journal", "year", "authors", "title")),
)

_ENTITY_HINT_RE = re.compile(
    r"\b(EGFR|ALK|ROS1|BRAF|KRAS|NRAS|MET|ERBB2|HER2|TP53|PIK3CA|RET|NTRK|VEGF|PD-?1|PD-?L1)\b",
    re.IGNORECASE,
)

_SECTION_HINTS: dict[QueryIntentType, tuple[str, ...]] = {
    QueryIntentType.SAMPLE_SIZE: ("methods", "results", "abstract"),
    QueryIntentType.CONCLUSIONS: ("discussion", "conclusion", "abstract"),
    QueryIntentType.METHODOLOGY: ("methods", "study design"),
    QueryIntentType.SAFETY: ("results", "safety", "adverse events"),
    QueryIntentType.EFFICACY: ("results", "abstract"),
}

_EXPANSIONS: dict[QueryIntentType, tuple[str, ...]] = {
    QueryIntentType.SAMPLE_SIZE: ("enrolled", "randomized", "screened", "patients", "assigned"),
    QueryIntentType.CONCLUSIONS: ("discussion", "conclusion", "interpretation"),
    QueryIntentType.METHODOLOGY: ("methods", "trial design", "eligibility"),
    QueryIntentType.MUTATION: ("gene", "protein", "variant", "pathway"),
}


class QueryUnderstandingEngine:
    """Rule-based biomedical query understanding with structured output."""

    def __init__(self, patterns: tuple[IdentifierPattern, ...] | None = None) -> None:
        self.patterns = patterns or DEFAULT_IDENTIFIER_PATTERNS
        self._compiled = [(item.type, item.compile()) for item in self.patterns]

    def understand(self, query: str, *, forced_filters: dict[str, Any] | None = None) -> QueryUnderstanding:
        cleaned = (query or "").strip()
        lowered = f" {cleaned.lower()} "
        identifiers = self._detect_identifiers(cleaned)
        intent = self._classify(lowered, identifiers)
        entities = tuple(dict.fromkeys(match.group(1).upper().replace("-", "") for match in _ENTITY_HINT_RE.finditer(cleaned)))
        filters: dict[str, Any] = {}
        if forced_filters:
            filters.update({key: value for key, value in forced_filters.items() if value is not None})
        if identifiers:
            filters["identifiers"] = [item.normalized for item in identifiers]
        if entities:
            filters["entities"] = list(entities)
        metadata = {
            "section_hints": list(_SECTION_HINTS.get(intent, ())),
            "document_reference": bool(re.search(r"\b(this|the)\s+(paper|study|trial|article)\b", lowered)),
        }
        expanded = self._expand(cleaned, intent, entities)
        return QueryUnderstanding(
            original_query=cleaned,
            expanded_query=expanded,
            intent=intent,
            identifiers=tuple(identifiers),
            entities=entities,
            filters=filters,
            metadata=metadata,
        )

    def _detect_identifiers(self, query: str) -> list[DetectedIdentifier]:
        detected: list[DetectedIdentifier] = []
        seen: set[tuple[str, str]] = set()
        for identifier_type, pattern in self._compiled:
            for match in pattern.finditer(query):
                raw = match.group(1) if match.groups() and match.group(1) else match.group(0)
                normalized = normalize_identifier(raw)
                key = (identifier_type, normalized)
                if key in seen:
                    continue
                seen.add(key)
                detected.append(
                    DetectedIdentifier(
                        type=identifier_type,
                        value=raw,
                        normalized=normalized,
                        span=(match.start(), match.end()),
                    )
                )
        detected.sort(key=lambda item: item.span)
        return detected

    @staticmethod
    def _classify(lowered_query: str, identifiers: list[DetectedIdentifier]) -> QueryIntentType:
        for intent, triggers in _INTENT_RULES:
            if any(trigger in lowered_query for trigger in triggers):
                return intent
        if _ENTITY_HINT_RE.search(lowered_query):
            return QueryIntentType.ENTITY
        if identifiers:
            return QueryIntentType.METADATA
        return QueryIntentType.GENERIC

    @staticmethod
    def _expand(query: str, intent: QueryIntentType, entities: tuple[str, ...]) -> str:
        additions = list(_EXPANSIONS.get(intent, ()))
        additions.extend(entities)
        if not additions:
            return query
        seen = {token.casefold() for token in query.split()}
        extras = [item for item in additions if item.casefold() not in seen]
        return f"{query} {' '.join(extras)}" if extras else query

"""Retrieval policy gatekeeper."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag_system.retrieval.query_understanding import QueryIntentType, QueryUnderstanding


class RetrievalPolicy(str, Enum):
    IDENTIFIER_DOCUMENT_SCOPE = "IDENTIFIER_DOCUMENT_SCOPE"
    MULTI_DOCUMENT_SCOPE = "MULTI_DOCUMENT_SCOPE"
    FORCED_DOCUMENT_SCOPE = "FORCED_DOCUMENT_SCOPE"
    DOCUMENT_SCOPE = "DOCUMENT_SCOPE"
    SECTION_SCOPE = "SECTION_SCOPE"
    SECTION_PRIORITY = "SECTION_PRIORITY"
    ENTITY_FIRST_RETRIEVAL = "ENTITY_FIRST_RETRIEVAL"
    GLOBAL_HYBRID_RETRIEVAL = "GLOBAL_HYBRID_RETRIEVAL"


@dataclass(frozen=True)
class RetrievalPlan:
    query: str
    identifiers: tuple[str, ...]
    policy: RetrievalPolicy
    matched_documents: tuple[str, ...] = ()
    scope: tuple[str, ...] = ()
    retrieval_strategy: str = "document_first"
    top_papers: tuple[str, ...] = ()
    top_sections: tuple[str, ...] = ()
    top_chunks: tuple[str, ...] = ()
    cross_encoder: str = "post_filter_only"
    evidence_builder: str = "parent_child_grouped"
    llm: str = "evidence_only"
    reasons: tuple[str, ...] = ()
    filters: dict[str, Any] = field(default_factory=dict)


_PAPER_ID_KEYS = {"paper_id", "paper_ids", "pmid", "pmcid", "trial_id", "trial_ids", "doi", "source", "document_id", "document_ids"}


class RetrievalPolicyEngine:
    """Mandatory decision tree that runs before retrieval."""

    def decide(
        self,
        understanding: QueryUnderstanding,
        *,
        matched_documents: set[str] | None = None,
        forced_filters: dict[str, Any] | None = None,
    ) -> RetrievalPlan:
        matched_documents = matched_documents or set()
        forced_filters = forced_filters or {}
        reasons: list[str] = []
        identifiers = tuple(item.normalized for item in understanding.identifiers)

        if any(key in forced_filters and forced_filters[key] for key in _PAPER_ID_KEYS):
            policy = RetrievalPolicy.FORCED_DOCUMENT_SCOPE
            reasons.append("evaluation_or_caller_forced_document_scope")
        elif len(identifiers) > 1:
            policy = RetrievalPolicy.MULTI_DOCUMENT_SCOPE
            reasons.append("multiple_identifiers_detected")
        elif identifiers:
            policy = RetrievalPolicy.IDENTIFIER_DOCUMENT_SCOPE
            reasons.append("identifier_detected")
        elif understanding.metadata.get("document_reference"):
            policy = RetrievalPolicy.DOCUMENT_SCOPE
            reasons.append("document_reference_phrase")
        elif understanding.intent == QueryIntentType.CONCLUSIONS:
            policy = RetrievalPolicy.SECTION_SCOPE
            reasons.append("conclusion_query")
        elif understanding.intent == QueryIntentType.SAMPLE_SIZE:
            policy = RetrievalPolicy.SECTION_PRIORITY
            reasons.append("sample_size_query")
        elif understanding.intent in {QueryIntentType.MUTATION, QueryIntentType.ENTITY} or understanding.entities:
            policy = RetrievalPolicy.ENTITY_FIRST_RETRIEVAL
            reasons.append("entity_query")
        else:
            policy = RetrievalPolicy.GLOBAL_HYBRID_RETRIEVAL
            reasons.append("default_global_hybrid")

        section_hints = tuple(understanding.metadata.get("section_hints", ()))
        if policy == RetrievalPolicy.SECTION_SCOPE and not section_hints:
            section_hints = ("discussion", "conclusion")
        elif policy == RetrievalPolicy.SECTION_PRIORITY and not section_hints:
            section_hints = ("methods", "results")

        scope = tuple(sorted(matched_documents))
        return RetrievalPlan(
            query=understanding.expanded_query,
            identifiers=identifiers,
            policy=policy,
            matched_documents=scope,
            scope=scope,
            top_sections=section_hints,
            reasons=tuple(reasons),
            filters={**understanding.filters, **forced_filters},
        )


def document_reference_in_query(query: str) -> bool:
    return bool(re.search(r"\b(this|the)\s+(paper|study|trial|article)\b", query or "", flags=re.IGNORECASE))

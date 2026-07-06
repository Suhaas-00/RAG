"""Document-first retrieval pipeline.

Pipeline:
Query -> Query Understanding -> Identifier Detection -> Identifier Resolution
-> Document Routing -> Section Retrieval -> Chunk Retrieval -> Reranker
-> Evidence Aggregation -> LLM context.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import faiss
import numpy as np

from rag_system.retrieval.document_registry import DocumentRegistry
from rag_system.retrieval.evidence import build_parent_child_evidence
from rag_system.retrieval.identifier_patterns import normalize_identifier
from rag_system.retrieval.policy_engine import RetrievalPlan, RetrievalPolicy, RetrievalPolicyEngine
from rag_system.retrieval.query_understanding import QueryUnderstanding, QueryUnderstandingEngine
from rag_system.utils.preprocessing import normalize_for_embedding

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RejectedChunk:
    chunk_id: str
    source: str | None
    document_id: str | None
    reason: str


@dataclass(frozen=True)
class QueryTrace:
    identifier_detected: tuple[str, ...] = ()
    policy: str = ""
    matched_documents: tuple[str, ...] = ()
    candidate_sections: tuple[str, ...] = ()
    retrieved_chunks: tuple[str, ...] = ()
    rejected_chunks: tuple[RejectedChunk, ...] = ()
    latencies_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentFirstResult:
    context: str
    chunks: list[dict[str, Any]]
    plan: RetrievalPlan
    trace: QueryTrace


class DocumentFirstRetriever:
    """Mandatory retrieval orchestrator for biomedical RAG."""

    def __init__(
        self,
        *,
        records: list[dict[str, Any]],
        index: faiss.Index,
        embedder: Any,
        payload: dict[str, Any],
        reranker: Any | None = None,
        registry: DocumentRegistry | None = None,
        query_engine: QueryUnderstandingEngine | None = None,
        policy_engine: RetrievalPolicyEngine | None = None,
        alpha: float = 0.55,
    ) -> None:
        self.records = records
        self.index = index
        self.embedder = embedder
        self.payload = payload
        self.reranker = reranker
        self.registry = registry or DocumentRegistry.from_payload(payload)
        self.query_engine = query_engine or QueryUnderstandingEngine()
        self.policy_engine = policy_engine or RetrievalPolicyEngine()
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 20,
        candidate_k: int = 80,
        alpha: float | None = None,
        forced_filters: dict[str, Any] | None = None,
        allow_global_search: bool | None = None,
    ) -> DocumentFirstResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        understanding = self.query_engine.understand(query, forced_filters=forced_filters)
        timings["identifier_detection"] = self._elapsed(t0)

        t1 = time.perf_counter()
        matched_documents = self._resolve_documents(understanding, forced_filters or {})
        timings["identifier_resolution"] = self._elapsed(t1)

        plan = self.policy_engine.decide(
            understanding,
            matched_documents=matched_documents,
            forced_filters=forced_filters,
        )
        strict_scope = self._is_strict_scope(plan)
        if strict_scope and not plan.scope:
            trace = QueryTrace(
                identifier_detected=tuple(item.normalized for item in understanding.identifiers),
                policy=plan.policy.value,
                matched_documents=(),
                candidate_sections=tuple(plan.top_sections),
                retrieved_chunks=(),
                rejected_chunks=(),
                latencies_ms=timings,
            )
            logger.warning("Strict document scope had no matched documents: %s", asdict(plan))
            return DocumentFirstResult("", [], plan, trace)

        t2 = time.perf_counter()
        candidate_indices = self._route_candidates(plan, understanding, allow_global_search=allow_global_search)
        section_indices = self._section_candidates(plan, candidate_indices)
        timings["document_routing"] = self._elapsed(t2)

        t3 = time.perf_counter()
        scored = self._score(understanding.expanded_query, section_indices, candidate_k, alpha or self.alpha)
        timings["chunk_retrieval"] = self._elapsed(t3)

        t4 = time.perf_counter()
        valid, rejected = self._enforce_scope(scored, plan, understanding)
        reranked = self._rerank(understanding.original_query, valid, top_k)
        timings["rerank"] = self._elapsed(t4)

        evidence = build_parent_child_evidence(reranked)
        retrieved_ids = tuple(str(row.get("chunk_id")) for row in reranked)
        plan = RetrievalPlan(
            query=plan.query,
            identifiers=plan.identifiers,
            policy=plan.policy,
            matched_documents=plan.matched_documents,
            scope=plan.scope,
            retrieval_strategy=plan.retrieval_strategy,
            top_papers=tuple(self.registry.documents[doc_id].source for doc_id in plan.scope if doc_id in self.registry.documents),
            top_sections=plan.top_sections,
            top_chunks=retrieved_ids,
            cross_encoder=plan.cross_encoder,
            evidence_builder=plan.evidence_builder,
            llm=plan.llm,
            reasons=plan.reasons,
            filters=plan.filters,
        )
        trace = QueryTrace(
            identifier_detected=tuple(item.normalized for item in understanding.identifiers),
            policy=plan.policy.value,
            matched_documents=plan.matched_documents,
            candidate_sections=tuple(plan.top_sections),
            retrieved_chunks=retrieved_ids,
            rejected_chunks=tuple(rejected),
            latencies_ms=timings,
        )
        logger.info("Retrieval decision: %s", asdict(trace))
        return DocumentFirstResult(evidence.context, reranked, plan, trace)

    def _resolve_documents(self, understanding: QueryUnderstanding, forced_filters: dict[str, Any]) -> set[str]:
        docs: set[str] = set()
        for identifier in understanding.identifiers:
            docs.update(self.registry.resolve(identifier.normalized))
        for key in ("paper_id", "document_id", "source", "pmid", "pmcid", "doi", "trial_id"):
            raw = forced_filters.get(key)
            if raw:
                values = raw if isinstance(raw, (list, tuple, set)) else [raw]
                for value in values:
                    docs.update(self.registry.resolve(str(value)))
        for key in ("paper_ids", "document_ids", "sources", "trial_ids"):
            for value in forced_filters.get(key, []) or []:
                docs.update(self.registry.resolve(str(value)))
        return docs

    @staticmethod
    def _is_strict_scope(plan: RetrievalPlan) -> bool:
        return plan.policy in {
            RetrievalPolicy.IDENTIFIER_DOCUMENT_SCOPE,
            RetrievalPolicy.MULTI_DOCUMENT_SCOPE,
            RetrievalPolicy.FORCED_DOCUMENT_SCOPE,
            RetrievalPolicy.DOCUMENT_SCOPE,
        }

    def _route_candidates(
        self,
        plan: RetrievalPlan,
        understanding: QueryUnderstanding,
        *,
        allow_global_search: bool | None,
    ) -> list[int]:
        if plan.scope:
            return sorted(self.registry.allowed_indices(plan.scope))
        if self._is_strict_scope(plan) and allow_global_search is not True:
            return []
        if plan.policy == RetrievalPolicy.ENTITY_FIRST_RETRIEVAL and understanding.entities:
            entity_terms = {item.casefold() for item in understanding.entities}
            hits = [
                idx for idx, record in enumerate(self.records)
                if any(term in self._record_blob(record).casefold() for term in entity_terms)
            ]
            if hits:
                return hits
        return list(range(len(self.records)))

    def _section_candidates(self, plan: RetrievalPlan, candidate_indices: list[int]) -> list[int]:
        if not candidate_indices:
            return []
        allowed = set(candidate_indices)
        if plan.scope and plan.top_sections:
            scoped_sections = [idx for idx in self.registry.section_indices(plan.scope, plan.top_sections) if idx in allowed]
            if scoped_sections:
                return scoped_sections
        if plan.top_sections:
            wanted = {section.casefold() for section in plan.top_sections}
            prioritized = [idx for idx in candidate_indices if str(self.records[idx].get("section", "")).casefold() in wanted]
            return prioritized + [idx for idx in candidate_indices if idx not in set(prioritized)]
        return candidate_indices

    def _score(self, query: str, candidate_indices: list[int], candidate_k: int, alpha: float) -> list[dict[str, Any]]:
        if not candidate_indices:
            return []
        query_vec = self.embedder.encode_query(query).astype("float32")
        vectors = np.vstack([self.index.reconstruct(int(idx)) for idx in candidate_indices]).astype("float32")
        sub_index = faiss.IndexFlatIP(self.index.d)
        sub_index.add(vectors)
        scores, local_ids = sub_index.search(query_vec, min(max(candidate_k, 1), len(candidate_indices)))
        dense: dict[int, float] = {
            candidate_indices[int(local)]: float(score)
            for local, score in zip(local_ids[0], scores[0])
            if local >= 0
        }
        lexical = self._lexical_scores(query, list(dense))
        rows: list[dict[str, Any]] = []
        for idx, dense_score in dense.items():
            record = dict(self.records[idx])
            lexical_score = lexical.get(idx, 0.0)
            hybrid = alpha * dense_score + (1.0 - alpha) * lexical_score
            record.update(
                {
                    "_position": idx,
                    "dense_score": round(dense_score, 4),
                    "bm25_score": round(lexical_score, 4),
                    "hybrid_score": round(hybrid, 4),
                    "final_score": round(hybrid, 4),
                }
            )
            rows.append(record)
        rows.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return rows

    def _lexical_scores(self, query: str, candidate_indices: list[int]) -> dict[int, float]:
        terms = re.findall(r"[a-z0-9][a-z0-9-]{1,}", normalize_for_embedding(query))
        if not terms:
            return {}
        docs = [re.findall(r"[a-z0-9][a-z0-9-]{1,}", normalize_for_embedding(self.records[idx].get("text", ""))) for idx in candidate_indices]
        if not docs:
            return {}
        avgdl = sum(len(doc) for doc in docs) / max(len(docs), 1)
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        raw: dict[int, float] = {}
        for idx, doc in zip(candidate_indices, docs):
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1 + (len(docs) - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                score += idf * (freq * 2.5) / (freq + 1.5 * (1 - 0.75 + 0.75 * len(doc) / max(avgdl, 1e-9)))
            if score:
                raw[idx] = score
        max_score = max(raw.values(), default=1.0)
        return {idx: value / max_score for idx, value in raw.items()}

    def _enforce_scope(
        self,
        rows: list[dict[str, Any]],
        plan: RetrievalPlan,
        understanding: QueryUnderstanding,
    ) -> tuple[list[dict[str, Any]], list[RejectedChunk]]:
        if not self._is_strict_scope(plan) or not plan.scope:
            return rows, []
        allowed_indices = self.registry.allowed_indices(plan.scope)
        allowed_documents = set(plan.scope)
        identifiers = {item.normalized for item in understanding.identifiers}
        valid: list[dict[str, Any]] = []
        rejected: list[RejectedChunk] = []
        for row in rows:
            idx = int(row.get("_position", -1))
            document_id = self._row_document_id(row)
            if idx not in allowed_indices and document_id not in allowed_documents:
                rejected.append(self._reject(row, "outside_document_scope"))
                continue
            if identifiers and not self._document_contains_identifiers(document_id, identifiers):
                rejected.append(self._reject(row, "document_missing_identifier_metadata"))
                continue
            valid.append(row)
        return valid, rejected

    def _rerank(self, query: str, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not rows:
            return []
        limit = max(top_k, 1)
        if self.reranker is not None:
            reranked = self.reranker.rerank(query, rows, top_k=limit)
        else:
            reranked = rows[:limit]
        for row in reranked:
            row["final_score"] = round(float(row.get("rerank_score", row.get("hybrid_score", 0.0)) or 0.0), 4)
            row.setdefault("confidence", min(max(float(row.get("hybrid_score", 0.0) or 0.0), 0.0), 1.0))
        return reranked[:limit]

    def _row_document_id(self, row: dict[str, Any]) -> str:
        return str(row.get("document_id") or row.get("paper_id") or row.get("source") or "")

    def _document_contains_identifiers(self, document_id: str, identifiers: set[str]) -> bool:
        if document_id in self.registry.documents:
            aliases = {normalize_identifier(alias) for alias in self.registry.documents[document_id].aliases}
            return bool(aliases & identifiers) or all(document_id in self.registry.resolve(identifier) for identifier in identifiers)
        return False

    @staticmethod
    def _reject(row: dict[str, Any], reason: str) -> RejectedChunk:
        return RejectedChunk(
            chunk_id=str(row.get("chunk_id") or ""),
            source=row.get("source"),
            document_id=str(row.get("document_id") or row.get("paper_id") or row.get("source") or ""),
            reason=reason,
        )

    @staticmethod
    def _record_blob(record: dict[str, Any]) -> str:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        return " ".join([str(record.get("text", "")), str(record.get("section", "")), " ".join(map(str, metadata.values()))])

    @staticmethod
    def _elapsed(start: float) -> float:
        return round((time.perf_counter() - start) * 1000.0, 3)

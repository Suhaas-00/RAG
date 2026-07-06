"""Application service layer for retrieval-augmented generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_system.cache import CacheManager, stable_cache_key
from rag_system.exceptions import IndexUnavailableError
from rag_system.hybrid_retriever import HybridRetriever, RetrievalResult
from rag_system.llm import FALLBACK, answer_with_groq, load_groq_api_key
from rag_system.query_parser import QueryIntent, build_retrieval_query, parse_query
from rag_system.telemetry import MetricsRegistry, traced_operation
from rag_system.utils.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerRequest:
    """Request model for a RAG answer."""

    question: str
    model: str = "llama-3.1-8b-instant"
    top_k: int = 5
    alpha: float = 0.55
    allow_global_search: bool | None = None
    use_cache: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class AnswerResponse:
    """Response model returned by :class:`RAGService`."""

    answer: str
    context: str
    chunks: list[dict[str, Any]]
    intent: str
    retrieval_query: str
    debug: dict[str, Any] = field(default_factory=dict)


class RAGService:
    """Dependency-injected application boundary for CLI, API, and jobs."""

    def __init__(
        self,
        *,
        settings: Settings,
        retriever: HybridRetriever,
        cache_manager: CacheManager | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.cache_manager = cache_manager or CacheManager(
            enabled=settings.enable_cache,
            max_size=settings.cache_max_size,
            retrieval_ttl_seconds=settings.retrieval_cache_ttl_seconds,
            response_ttl_seconds=settings.response_cache_ttl_seconds,
        )
        self.metrics = metrics or MetricsRegistry()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGService":
        """Create a service by loading configured index artifacts."""

        settings = settings or Settings.from_env()
        if not settings.vectors_path.exists() or not settings.metadata_path.exists():
            raise IndexUnavailableError(
                f"Index artifacts not found under {settings.index_dir}. "
                "Run ingestion before starting the service."
            )
        retriever = HybridRetriever.load(
            settings.index_dir,
            alpha=settings.retrieval_alpha,
            enable_document_filtering=settings.enable_document_filtering,
            allow_global_search=settings.allow_global_search,
        )
        return cls(settings=settings, retriever=retriever)

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        """Answer one natural-language question."""

        with traced_operation(self.metrics, "rag.answer"):
            if not request.question.strip():
                return AnswerResponse(
                    answer=FALLBACK,
                    context="",
                    chunks=[],
                    intent="empty",
                    retrieval_query="",
                    debug={"reason": "empty_question"},
                )

            paper_map = {label: filename for label, filename in self.retriever.list_papers()}
            intent = parse_query(request.question, paper_map)
            retrieval_query = build_retrieval_query(intent)

            response_key = stable_cache_key(
                "response",
                {
                    "question": request.question,
                    "model": request.model,
                    "top_k": request.top_k,
                    "alpha": request.alpha,
                    "allow_global_search": request.allow_global_search,
                    "settings_model": self.settings.model_name,
                },
            )
            if self.cache_manager.enabled and request.use_cache:
                cached = self.cache_manager.response_cache.get(response_key)
                if isinstance(cached, AnswerResponse):
                    self.metrics.increment("rag.answer.cache_hit")
                    return cached

            if intent.intent in {"list_papers", "metadata_query"}:
                result = RetrievalResult(context="", chunks=[], debug={"short_circuit": intent.intent})
                answer = self._generate_answer(request, intent, result)
            else:
                result = self._retrieve(request, intent, retrieval_query)
                answer = self._generate_answer(request, intent, result)
            response = AnswerResponse(
                answer=answer,
                context=result.context,
                chunks=result.chunks,
                intent=intent.intent,
                retrieval_query=retrieval_query,
                debug=result.debug,
            )
            if self.cache_manager.enabled and request.use_cache:
                self.cache_manager.response_cache.set(response_key, response)
            return response

    def list_papers(self) -> list[tuple[str, str]]:
        """Return configured paper labels and source filenames."""

        return self.retriever.list_papers()

    def health(self) -> dict[str, Any]:
        """Return process-local health and readiness information."""

        artifacts = {
            "index_dir": str(self.settings.index_dir),
            "vectors_exists": self.settings.vectors_path.exists(),
            "metadata_exists": self.settings.metadata_path.exists(),
        }
        metrics = self.metrics.snapshot()
        return {
            "status": "ok" if all(artifacts[key] for key in ("vectors_exists", "metadata_exists")) else "degraded",
            "artifacts": artifacts,
            "records": len(self.retriever.records),
            "cache": self.cache_manager.stats(),
            "metrics": {
                "counters": metrics.counters,
                "latencies_ms": metrics.latencies_ms,
            },
        }

    def _retrieve(
        self,
        request: AnswerRequest,
        intent: QueryIntent,
        retrieval_query: str,
    ) -> RetrievalResult:
        cache_key = stable_cache_key(
            "retrieval",
            {
                "query": retrieval_query,
                "intent": intent.intent,
                "filters": intent.filters,
                "top_k": request.top_k,
                "alpha": request.alpha,
                "allow_global_search": request.allow_global_search,
            },
        )
        if self.cache_manager.enabled and request.use_cache:
            cached = self.cache_manager.retrieval_cache.get(cache_key)
            if isinstance(cached, RetrievalResult):
                self.metrics.increment("rag.retrieval.cache_hit")
                return cached

        with traced_operation(self.metrics, "rag.retrieval", attributes={"intent": intent.intent}):
            result = self.retriever.retrieve(
                retrieval_query,
                intent,
                top_k=request.top_k,
                alpha=request.alpha,
                allow_global_search=request.allow_global_search,
            )
        if self.cache_manager.enabled and request.use_cache:
            self.cache_manager.retrieval_cache.set(cache_key, result)
        return result

    def _generate_answer(
        self,
        request: AnswerRequest,
        intent: QueryIntent,
        result: RetrievalResult,
    ) -> str:
        if intent.intent == "list_papers":
            rows = self.retriever.list_papers()
            return "\n".join(f"{label}: {filename}" for label, filename in rows) if rows else FALLBACK
        if intent.intent == "metadata_query":
            return self._metadata_answer(intent)
        if not result.chunks:
            return FALLBACK
        if intent.intent in {"paper_lookup", "section_lookup"} and not load_groq_api_key(Path.cwd()):
            return self._format_direct_section(result)
        with traced_operation(self.metrics, "rag.generation", attributes={"model": request.model}):
            return answer_with_groq(request.question, result.context, model=request.model)

    def _metadata_answer(self, intent: QueryIntent) -> str:
        if not intent.paper_source and not intent.paper_label:
            return FALLBACK
        keys = [intent.paper_source, Path(intent.paper_source or "").stem, intent.paper_label]
        metadata = None
        for key in keys:
            if key and key in self.retriever.payload.get("doc_metadata_index", {}):
                metadata = self.retriever.payload["doc_metadata_index"][key]
                break
        if not metadata:
            return FALLBACK
        field_name = intent.metadata_field
        if field_name and metadata.get(field_name):
            return (
                f"{field_name.upper()}: {metadata[field_name]} "
                f"(Source: {metadata.get('source')}; Section: metadata; Page: 1; Confidence: 1.0)"
            )
        lines = [f"Metadata for {metadata.get('paper_id', intent.paper_label)} / {metadata.get('source')}:"]
        for key in ("doi", "pmid", "year", "journal"):
            if metadata.get(key):
                lines.append(f"- {key.upper()}: {metadata[key]}")
        for key, label in (("diseases", "Diseases"), ("genes", "Genes"), ("study_designs", "Study designs")):
            if metadata.get(key):
                lines.append(f"- {label}: {', '.join(metadata[key])}")
        lines.append(f"(Source: {metadata.get('source')}; Section: metadata; Page: 1; Confidence: 1.0)")
        return "\n".join(lines)

    @staticmethod
    def _format_direct_section(result: RetrievalResult) -> str:
        lines: list[str] = []
        for chunk in result.chunks:
            page = chunk.get("page", chunk.get("page_number", "?"))
            lines.append(
                f"Source: {chunk.get('source')} | Paper: {chunk.get('paper_id')} | "
                f"Section: {chunk.get('section')} | Page: {page} | "
                f"Confidence: {chunk.get('confidence', '?')}\n{chunk.get('text', '')}"
            )
        return "\n\n".join(lines) if lines else FALLBACK

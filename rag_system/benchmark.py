"""Simple benchmark harness for RAG service latency."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Iterable

from rag_system.service import AnswerRequest, RAGService


@dataclass(frozen=True)
class BenchmarkResult:
    """Latency summary for a benchmark run."""

    requests: int
    average_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


def benchmark_questions(
    service: RAGService,
    questions: Iterable[str],
    *,
    top_k: int = 5,
    use_cache: bool = False,
) -> BenchmarkResult:
    """Measure end-to-end service latency for a question iterable."""

    latencies: list[float] = []
    for question in questions:
        start = time.perf_counter()
        service.answer(AnswerRequest(question=question, top_k=top_k, use_cache=use_cache))
        latencies.append((time.perf_counter() - start) * 1000.0)
    if not latencies:
        return BenchmarkResult(requests=0, average_ms=0.0, p50_ms=0.0, p95_ms=0.0, max_ms=0.0)
    ordered = sorted(latencies)
    return BenchmarkResult(
        requests=len(ordered),
        average_ms=statistics.fmean(ordered),
        p50_ms=ordered[int((len(ordered) - 1) * 0.50)],
        p95_ms=ordered[int((len(ordered) - 1) * 0.95)],
        max_ms=ordered[-1],
    )


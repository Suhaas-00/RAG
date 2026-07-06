"""Evaluation utilities for retrieval and answer quality regression checks."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rag_system.service import AnswerRequest, RAGService


@dataclass(frozen=True)
class EvaluationCase:
    """Single evaluation row."""

    question: str
    expected_sources: tuple[str, ...] = ()
    expected_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    """Per-question evaluation output."""

    question: str
    source_recall: float
    term_coverage: float
    retrieved_sources: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation metrics."""

    cases: int
    mean_source_recall: float
    mean_term_coverage: float
    results: tuple[EvaluationResult, ...]


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load evaluation cases from JSONL or CSV."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        cases: list[EvaluationCase] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                cases.append(
                    EvaluationCase(
                        question=str(data["question"]),
                        expected_sources=tuple(data.get("expected_sources", ())),
                        expected_terms=tuple(data.get("expected_terms", ())),
                    )
                )
        return cases

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            EvaluationCase(
                question=row["question"],
                expected_sources=tuple(_split_cell(row.get("expected_sources", ""))),
                expected_terms=tuple(_split_cell(row.get("expected_terms", ""))),
            )
            for row in reader
        ]


def evaluate(service: RAGService, cases: Iterable[EvaluationCase], *, top_k: int = 5) -> EvaluationSummary:
    """Run deterministic retrieval-oriented evaluation cases."""

    results: list[EvaluationResult] = []
    for case in cases:
        response = service.answer(AnswerRequest(question=case.question, top_k=top_k, use_cache=False))
        retrieved_sources = tuple(
            str(chunk.get("source", ""))
            for chunk in response.chunks
            if chunk.get("source")
        )
        results.append(
            EvaluationResult(
                question=case.question,
                source_recall=_source_recall(case.expected_sources, retrieved_sources),
                term_coverage=_term_coverage(case.expected_terms, response.answer),
                retrieved_sources=retrieved_sources,
                answer=response.answer,
            )
        )
    count = len(results)
    return EvaluationSummary(
        cases=count,
        mean_source_recall=sum(item.source_recall for item in results) / max(count, 1),
        mean_term_coverage=sum(item.term_coverage for item in results) / max(count, 1),
        results=tuple(results),
    )


def save_evaluation_summary(summary: EvaluationSummary, path: str | Path) -> None:
    """Write an evaluation summary as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, ensure_ascii=False)


def _split_cell(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _source_recall(expected: tuple[str, ...], retrieved: tuple[str, ...]) -> float:
    if not expected:
        return 1.0
    retrieved_fold = {item.casefold() for item in retrieved}
    hits = sum(1 for item in expected if item.casefold() in retrieved_fold)
    return hits / len(expected)


def _term_coverage(expected_terms: tuple[str, ...], answer: str) -> float:
    if not expected_terms:
        return 1.0
    answer_fold = answer.casefold()
    hits = sum(1 for term in expected_terms if term.casefold() in answer_fold)
    return hits / len(expected_terms)


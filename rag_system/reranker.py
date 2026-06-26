"""CrossEncoder reranking with deterministic fallback scoring."""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            logger.info("Loaded CrossEncoder reranker: %s", model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossEncoder unavailable; using hybrid scores only: %s", exc)

    def rerank(self, query: str, candidates: Iterable[dict], top_k: int = 5) -> list[dict]:
        rows = [dict(candidate) for candidate in candidates]
        if not rows:
            return []

        if self.model is not None:
            pairs = [(query, row.get("text", "")) for row in rows]
            scores = self.model.predict(pairs)
            for row, score in zip(rows, scores):
                row["rerank_score"] = float(score)
                row["confidence"] = _confidence(float(score), row.get("hybrid_score", row.get("final_score", 0.0)))
        else:
            for row in rows:
                score = float(row.get("hybrid_score", row.get("final_score", 0.0)))
                row["rerank_score"] = score
                row["confidence"] = _confidence(score, score)

        rows.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return rows[:top_k]


def _confidence(rerank_score: float, hybrid_score: float) -> float:
    if rerank_score > 1:
        rerank_component = min(rerank_score / 10.0, 1.0)
    else:
        rerank_component = max(0.0, min((rerank_score + 1.0) / 2.0, 1.0))
    hybrid_component = max(0.0, min(float(hybrid_score), 1.0))
    return round(0.65 * rerank_component + 0.35 * hybrid_component, 3)


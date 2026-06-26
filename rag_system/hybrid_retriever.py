"""Metadata-first hybrid retrieval: filters, BM25/TF-IDF, FAISS, reranking."""

from __future__ import annotations

import logging
import pickle
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from rag_system.ingestion.embedding import PubMedEmbedder
from rag_system.query_parser import QueryIntent
from rag_system.reranker import CrossEncoderReranker
from rag_system.utils.preprocessing import normalize_for_embedding

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    context: str
    chunks: list[dict] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, index: faiss.Index, payload: dict, embedder: PubMedEmbedder, *, alpha: float = 0.55) -> None:
        self.index = index
        self.payload = payload
        self.embedder = embedder
        self.alpha = alpha
        self.records: list[dict] = payload["records"]
        self.tfidf = payload.get("tfidf_vectorizer")
        self.tfidf_matrix = payload.get("tfidf_matrix")
        self.paper_map: dict[str, str] = payload.get("paper_map", {})
        self.reranker = CrossEncoderReranker()

    @classmethod
    def load(cls, index_dir: str | Path, *, alpha: float = 0.55) -> "HybridRetriever":
        index_dir = Path(index_dir)
        index_path = index_dir / "vectors.index"
        metadata_path = index_dir / "metadata.pkl"
        if not index_path.exists():
            legacy = index_dir / "index.faiss"
            index_path = legacy if legacy.exists() else index_path
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing index files in {index_dir}")
        index = faiss.read_index(str(index_path))
        with metadata_path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301
        embedder = PubMedEmbedder(payload.get("model_name", "NeuML/pubmedbert-base-embeddings"), use_prefixes=True)
        return cls(index, payload, embedder, alpha=alpha)

    def retrieve(self, query: str, intent: QueryIntent | None = None, *, top_k: int = 5, candidate_k: int = 40, alpha: float | None = None, **legacy_filters) -> RetrievalResult:
        alpha = self.alpha if alpha is None else alpha
        filters = dict(getattr(intent, "filters", {}) or {})
        if legacy_filters.get("source_filter"):
            filters["source"] = legacy_filters["source_filter"]
        if legacy_filters.get("section_filter"):
            filters["section"] = legacy_filters["section_filter"]
        if legacy_filters.get("chunk_type_filter"):
            filters["chunk_type"] = legacy_filters["chunk_type_filter"]

        candidate_indices = self._metadata_filter(filters)
        debug = {"filters": filters, "candidate_count": len(candidate_indices)}
        if not candidate_indices:
            return RetrievalResult("", [], debug)

        dense_scores = self._dense_scores(query, candidate_indices, candidate_k)
        lexical_scores = self._lexical_scores(query, candidate_indices)

        all_indices = set(dense_scores) | set(lexical_scores)
        scored: list[dict] = []
        for idx in all_indices:
            dense = dense_scores.get(idx, 0.0)
            lexical = lexical_scores.get(idx, 0.0)
            hybrid = alpha * dense + (1.0 - alpha) * lexical
            record = dict(self.records[idx])
            record["dense_score"] = round(dense, 4)
            record["bm25_score"] = round(lexical, 4)
            record["hybrid_score"] = round(hybrid, 4)
            record["final_score"] = record["hybrid_score"]
            scored.append(record)

        scored.sort(key=lambda item: item["hybrid_score"], reverse=True)
        reranked = self.reranker.rerank(query, scored[:candidate_k], top_k=top_k)
        for row in reranked:
            row["final_score"] = round(float(row.get("rerank_score", row.get("hybrid_score", 0.0))), 4)
            row.setdefault("confidence", min(max(float(row.get("hybrid_score", 0.0)), 0.0), 1.0))

        debug["retrieved_chunks"] = [
            {
                "chunk_id": r.get("chunk_id"),
                "source": r.get("source"),
                "paper_id": r.get("paper_id"),
                "section": r.get("section"),
                "page": r.get("page", r.get("page_number")),
                "dense": r.get("dense_score"),
                "bm25": r.get("bm25_score"),
                "rerank": r.get("rerank_score"),
                "confidence": r.get("confidence"),
            }
            for r in reranked
        ]
        return RetrievalResult(format_context(reranked), reranked, debug)

    def _metadata_filter(self, filters: dict[str, str]) -> list[int]:
        result: list[int] = []
        for idx, rec in enumerate(self.records):
            if filters.get("source") and Path(str(rec.get("source", ""))).name.casefold() != Path(filters["source"]).name.casefold():
                continue
            if filters.get("paper_id") and str(rec.get("paper_id", "")).casefold() != filters["paper_id"].casefold():
                continue
            if filters.get("section") and str(rec.get("section", "")).casefold() != filters["section"].casefold():
                continue
            if filters.get("chunk_type") and str(rec.get("chunk_type", "content")) != filters["chunk_type"]:
                continue
            result.append(idx)
        return result

    def _dense_scores(self, query: str, candidate_indices: list[int], candidate_k: int) -> dict[int, float]:
        query_vec = self.embedder.encode_query(query).astype("float32")
        vectors = np.vstack([self.index.reconstruct(int(idx)) for idx in candidate_indices]).astype("float32")
        sub_index = faiss.IndexFlatIP(self.index.d)
        sub_index.add(vectors)
        scores, local_ids = sub_index.search(query_vec, min(candidate_k, len(candidate_indices)))
        return {
            candidate_indices[int(local)]: float(score)
            for local, score in zip(local_ids[0], scores[0])
            if local >= 0
        }

    def _lexical_scores(self, query: str, candidate_indices: list[int]) -> dict[int, float]:
        query_terms = _tokens(normalize_for_embedding(query))
        if not query_terms:
            return {}
        docs = [_tokens(self.records[idx].get("text", "")) for idx in candidate_indices]
        if not docs:
            return {}

        avgdl = sum(len(doc) for doc in docs) / max(len(docs), 1)
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        raw_scores: dict[int, float] = {}
        k1 = 1.5
        b = 0.75
        total_docs = len(docs)
        for idx, doc in zip(candidate_indices, docs):
            if not doc:
                continue
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1 + (total_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                denom = freq + k1 * (1 - b + b * len(doc) / max(avgdl, 1e-9))
                score += idf * (freq * (k1 + 1)) / denom
            if score > 0:
                raw_scores[idx] = score

        max_score = max(raw_scores.values(), default=1.0)
        return {idx: score / max_score for idx, score in raw_scores.items()}

    def list_papers(self) -> list[tuple[str, str]]:
        if self.paper_map:
            return sorted(self.paper_map.items(), key=lambda item: int(item[0].split()[-1]))
        sources = sorted({str(rec.get("source")) for rec in self.records if rec.get("source")})
        return [(f"paper {i}", source) for i, source in enumerate(sources, 1)]


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for rec in chunks:
        page = rec.get("page", rec.get("page_number", "?"))
        confidence = rec.get("confidence", "?")
        header = (
            f"[Source: {rec.get('source', '?')} | Paper: {rec.get('paper_id', '?')} | "
            f"Section: {rec.get('section', '?')} | Page: {page} | Confidence: {confidence}]"
        )
        parts.append(f"{header}\n{rec.get('text', '')}")
    return "\n\n".join(parts)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9-]{1,}", normalize_for_embedding(text))

from __future__ import annotations

import unittest

import faiss
import numpy as np

from rag_system.hybrid_retriever import HybridRetriever
from rag_system.metadata import normalize_record
from rag_system.retrieval.document_scope import filter_record_indices


class DummyEmbedder:
    dimension = 2

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype="float32")


class DummyReranker:
    def rerank(self, query: str, candidates, top_k: int = 5):
        return list(candidates)[:top_k]


def _index() -> faiss.Index:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    return index


def _records() -> list[dict]:
    return [
        {
            "chunk_id": "p1-a",
            "text": "egfr erlotinib response",
            "paper_id": "paper 1",
            "document_id": "paper 1",
            "source": "111.pdf",
            "section": "results",
            "page_number": 1,
            "chunk_type": "content",
            "metadata": {"paper_id": "paper 1", "document_id": "paper 1"},
        },
        {
            "chunk_id": "p1-b",
            "text": "egfr mutation survival",
            "paper_id": "paper 1",
            "document_id": "paper 1",
            "source": "111.pdf",
            "section": "discussion",
            "page_number": 2,
            "chunk_type": "content",
            "metadata": {"paper_id": "paper 1", "document_id": "paper 1"},
        },
        {
            "chunk_id": "p2-a",
            "text": "met tepotinib response",
            "paper_id": "paper 2",
            "document_id": "paper 2",
            "source": "222.pdf",
            "section": "results",
            "page_number": 1,
            "chunk_type": "content",
            "metadata": {"paper_id": "paper 2", "document_id": "paper 2"},
        },
        {
            "chunk_id": "legacy",
            "text": "legacy source only",
            "source": "333.pdf",
            "section": "results",
            "page_number": 1,
            "chunk_type": "content",
            "metadata": {"source": "333.pdf"},
        },
    ]


def _retriever() -> HybridRetriever:
    retriever = HybridRetriever(
        _index(),
        {"records": _records(), "paper_map": {"paper 1": "111.pdf", "paper 2": "222.pdf"}},
        DummyEmbedder(),
        enable_document_filtering=True,
        allow_global_search=False,
    )
    retriever.reranker = DummyReranker()
    return retriever


class DocumentScopeFilteringTests(unittest.TestCase):
    def test_missing_document_scope_returns_no_candidates_by_default(self) -> None:
        indices = filter_record_indices(_records(), {}, allow_global_search=False)
        self.assertEqual(indices, [])

    def test_global_search_is_explicit(self) -> None:
        indices = filter_record_indices(_records(), {}, allow_global_search=True)
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_single_document_active_retrieval_only_returns_requested_paper(self) -> None:
        result = _retriever().retrieve("response", paper_id_filter="paper 1", top_k=5)
        self.assertTrue(result.chunks)
        self.assertEqual({chunk.get("paper_id") for chunk in result.chunks}, {"paper 1"})

    def test_multiple_document_active_retrieval_only_returns_requested_papers(self) -> None:
        result = _retriever().retrieve(
            "response",
            paper_ids=["paper 1", "paper 2"],
            top_k=5,
        )
        self.assertTrue(result.chunks)
        self.assertLessEqual({chunk.get("paper_id") for chunk in result.chunks}, {"paper 1", "paper 2"})

    def test_legacy_source_only_record_can_still_be_scoped(self) -> None:
        indices = filter_record_indices(_records(), {"source": "333.pdf"}, allow_global_search=False)
        self.assertEqual(indices, [3])

    def test_normalize_record_adds_document_metadata_aliases(self) -> None:
        record = normalize_record(
            {
                "chunk_id": "abc",
                "source": "111.pdf",
                "page_number": 2,
                "section": "Results",
                "text": "sample",
            },
            paper_id="paper 1",
        )
        self.assertEqual(record["document_id"], "paper 1")
        self.assertEqual(record["filename"], "111.pdf")
        self.assertEqual(record["metadata"]["document_id"], "paper 1")
        self.assertEqual(record["metadata"]["chunk_id"], "abc")


if __name__ == "__main__":
    unittest.main()

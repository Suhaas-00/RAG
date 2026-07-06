from __future__ import annotations

import unittest

import faiss
import numpy as np

from rag_system.hybrid_retriever import HybridRetriever
from rag_system.retrieval.document_scope import filter_record_indices
from rag_system.retrieval.hybrid_search import hybrid_search
from rag_system.retrieval.retrieval_config import set_document_scope


class DummyEmbedder:
    dimension = 2

    def encode(self, texts, *args, **kwargs) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype="float32"), (len(texts), 1))

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype="float32")


class DummyReranker:
    def rerank(self, query: str, candidates, top_k: int = 5):
        return list(candidates)[:top_k]


def make_index() -> faiss.Index:
    vectors = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    return index


def make_records() -> list[dict]:
    return [
        {
            "chunk_id": "p1-results",
            "text": "egfr response",
            "paper_id": "paper 1",
            "document_id": "doc-1",
            "source": "111.pdf",
            "filename": "111.pdf",
            "section": "results",
            "page_number": 1,
            "chunk_type": "content",
            "metadata": {"diseases": ["lung cancer"], "genes": ["EGFR"], "year": "2020"},
        },
        {
            "chunk_id": "p1-methods",
            "text": "egfr methods",
            "paper_id": "paper 1",
            "document_id": "doc-1",
            "source": "111.pdf",
            "filename": "111.pdf",
            "section": "methods",
            "page_number": 2,
            "chunk_type": "content",
            "metadata": {"diseases": ["lung cancer"], "genes": ["EGFR"], "year": "2020"},
        },
        {
            "chunk_id": "p2-results",
            "text": "met response",
            "paper_id": "paper 2",
            "document_id": "doc-2",
            "source": "222.pdf",
            "filename": "222.pdf",
            "section": "results",
            "page_number": 1,
            "chunk_type": "content",
            "metadata": {"diseases": ["melanoma"], "genes": ["MET"], "year": "2021"},
        },
        {
            "chunk_id": "legacy-source",
            "text": "legacy response",
            "source": "legacy.pdf",
            "filename": "legacy.pdf",
            "section": "results",
            "page_number": 1,
            "chunk_type": "content",
            "metadata": {"diseases": ["legacy disease"], "year": "2019"},
        },
    ]


def make_retriever() -> HybridRetriever:
    retriever = HybridRetriever(
        make_index(),
        {"records": make_records(), "paper_map": {"paper 1": "111.pdf", "paper 2": "222.pdf"}},
        DummyEmbedder(),
    )
    retriever.reranker = DummyReranker()
    return retriever


class DocumentScopeConfigurationTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_document_scope(False)

    def test_global_retrieval_without_document_id_searches_entire_corpus(self) -> None:
        set_document_scope(True)
        result = make_retriever().retrieve("response", top_k=10)
        self.assertEqual(len(result.chunks), 4)

    def test_paper_id_retrieval_scopes_when_enabled(self) -> None:
        set_document_scope(True)
        result = make_retriever().retrieve("response", paper_id_filter="paper 1", top_k=10)
        self.assertTrue(result.chunks)
        self.assertEqual({chunk["paper_id"] for chunk in result.chunks}, {"paper 1"})

    def test_filename_retrieval_scopes_when_enabled(self) -> None:
        set_document_scope(True)
        result = make_retriever().retrieve("response", source_filter="222.pdf", top_k=10)
        self.assertTrue(result.chunks)
        self.assertEqual({chunk["source"] for chunk in result.chunks}, {"222.pdf"})

    def test_document_id_retrieval_scopes_when_enabled(self) -> None:
        set_document_scope(True)
        result = make_retriever().retrieve("response", document_id_filter="doc-2", top_k=10)
        self.assertTrue(result.chunks)
        self.assertEqual({chunk["document_id"] for chunk in result.chunks}, {"doc-2"})

    def test_section_filter_without_document_id_is_global(self) -> None:
        set_document_scope(True)
        indices = filter_record_indices(make_records(), {"section": "results"})
        self.assertEqual(indices, [0, 2, 3])

    def test_metadata_filters_still_work_globally(self) -> None:
        set_document_scope(False)
        indices = filter_record_indices(make_records(), {"gene": "MET"})
        self.assertEqual(indices, [2])

    def test_document_filtering_disabled_ignores_document_filters(self) -> None:
        set_document_scope(False)
        result = make_retriever().retrieve("response", paper_id_filter="paper 1", top_k=10)
        self.assertEqual(len(result.chunks), 4)

    def test_document_filtering_enabled_applies_document_filters(self) -> None:
        set_document_scope(True)
        indices = filter_record_indices(make_records(), {"source": "legacy.pdf"})
        self.assertEqual(indices, [3])

    def test_hybrid_search_global_default_without_document_id(self) -> None:
        set_document_scope(True)
        results = hybrid_search("response", DummyEmbedder(), make_index(), {"records": make_records()}, candidate_k=4)
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import time
import unittest

from rag_system.cache import CacheManager, TTLCache, stable_cache_key
from rag_system.hybrid_retriever import RetrievalResult
from rag_system.service import AnswerRequest, RAGService
from rag_system.telemetry import MetricsRegistry, traced_operation
from rag_system.utils.config import Settings


class DeterministicRetriever:
    payload = {"doc_metadata_index": {}}
    records = [{"source": "111.pdf"}]

    def __init__(self) -> None:
        self.retrieve_calls = 0

    def list_papers(self) -> list[tuple[str, str]]:
        return [("paper 1", "111.pdf")]

    def retrieve(self, query, intent=None, **kwargs) -> RetrievalResult:
        self.retrieve_calls += 1
        return RetrievalResult(
            context="[Source: 111.pdf | Paper: paper 1 | Section: abstract | Page: 1]\nEGFR response",
            chunks=[
                {
                    "chunk_id": "c1",
                    "source": "111.pdf",
                    "paper_id": "paper 1",
                    "section": "abstract",
                    "page": 1,
                    "confidence": 0.9,
                    "text": "EGFR response",
                }
            ],
            debug={"query": query, "kwargs": kwargs},
        )


class PlatformInfrastructureTests(unittest.TestCase):
    def test_ttl_cache_tracks_hits_misses_and_expiry(self) -> None:
        cache: TTLCache[str] = TTLCache(max_size=2, ttl_seconds=1)
        cache.set("a", "alpha")
        self.assertEqual(cache.get("a"), "alpha")
        self.assertIsNone(cache.get("b"))
        time.sleep(1.05)
        self.assertIsNone(cache.get("a"))
        stats = cache.stats()
        self.assertEqual(stats.hits, 1)
        self.assertGreaterEqual(stats.misses, 2)
        self.assertGreaterEqual(stats.evictions, 1)

    def test_cache_key_is_stable_for_equivalent_payloads(self) -> None:
        left = stable_cache_key("x", {"b": 2, "a": 1})
        right = stable_cache_key("x", {"a": 1, "b": 2})
        self.assertEqual(left, right)

    def test_settings_accepts_production_runtime_fields(self) -> None:
        settings = Settings.from_dict(
            {
                "retrieval_alpha": 0.6,
                "enable_cache": False,
                "cache_max_size": 8,
                "api_port": 9000,
                "log_level": "DEBUG",
            }
        )
        self.assertEqual(settings.retrieval_alpha, 0.6)
        self.assertFalse(settings.enable_cache)
        self.assertEqual(settings.cache_max_size, 8)
        self.assertEqual(settings.api_port, 9000)
        self.assertEqual(settings.log_level, "DEBUG")

    def test_service_short_circuits_list_papers_without_retrieval(self) -> None:
        retriever = DeterministicRetriever()
        service = RAGService(
            settings=Settings(enable_cache=False),
            retriever=retriever,
            cache_manager=CacheManager(enabled=False),
        )
        response = service.answer(AnswerRequest(question="list papers"))
        self.assertIn("paper 1: 111.pdf", response.answer)
        self.assertEqual(retriever.retrieve_calls, 0)
        self.assertEqual(response.intent, "list_papers")

    def test_metrics_registry_records_successful_operation(self) -> None:
        metrics = MetricsRegistry()
        with traced_operation(metrics, "unit.operation"):
            pass
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot.counters["unit.operation.started"], 1)
        self.assertEqual(snapshot.counters["unit.operation.completed"], 1)
        self.assertIn("unit.operation", snapshot.latencies_ms)


if __name__ == "__main__":
    unittest.main()


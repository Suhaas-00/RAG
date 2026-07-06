"""Lightweight telemetry primitives with no external backend requirement."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Current metric counters and latency summaries."""

    counters: dict[str, int] = field(default_factory=dict)
    latencies_ms: dict[str, dict[str, float]] = field(default_factory=dict)


class MetricsRegistry:
    """In-memory metric registry suitable for health endpoints and tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe_latency(self, name: str, milliseconds: float) -> None:
        with self._lock:
            values = self._latencies[name]
            values.append(milliseconds)
            if len(values) > 1000:
                del values[: len(values) - 1000]

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            latency_summary: dict[str, dict[str, float]] = {}
            for name, values in self._latencies.items():
                if not values:
                    continue
                ordered = sorted(values)
                count = len(ordered)
                latency_summary[name] = {
                    "count": float(count),
                    "avg": sum(ordered) / count,
                    "p50": ordered[int((count - 1) * 0.50)],
                    "p95": ordered[int((count - 1) * 0.95)],
                    "max": ordered[-1],
                }
            return MetricSnapshot(counters=dict(self._counters), latencies_ms=latency_summary)


@contextlib.contextmanager
def traced_operation(
    metrics: MetricsRegistry,
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record count, latency, and failure metrics around a block."""

    start = time.perf_counter()
    metrics.increment(f"{name}.started")
    try:
        yield
    except Exception:
        metrics.increment(f"{name}.failed")
        logger.exception("Operation failed", extra={"extra": {"operation": name, **(attributes or {})}})
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.increment(f"{name}.completed")
        metrics.observe_latency(name, elapsed_ms)


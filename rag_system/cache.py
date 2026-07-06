"""Thread-safe TTL caches for retrieval and response reuse."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CacheStats:
    """Point-in-time cache statistics."""

    hits: int
    misses: int
    evictions: int
    size: int
    max_size: int


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Small in-process TTL cache with bounded size and deterministic keys."""

    def __init__(self, *, max_size: int = 512, ttl_seconds: int = 300) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be >= 1, got {ttl_seconds}")
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, _Entry[T]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at <= now:
                self._items.pop(key, None)
                self._misses += 1
                self._evictions += 1
                return None
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T) -> None:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            if key not in self._items and len(self._items) >= self.max_size:
                oldest_key = min(self._items, key=lambda item_key: self._items[item_key].expires_at)
                self._items.pop(oldest_key, None)
                self._evictions += 1
            self._items[key] = _Entry(value=value, expires_at=now + self.ttl_seconds)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            self._purge_expired(time.monotonic())
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._items),
                max_size=self.max_size,
            )

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, entry in self._items.items() if entry.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)
        self._evictions += len(expired)


def stable_cache_key(namespace: str, payload: object) -> str:
    """Return a stable SHA-256 key for JSON-serialisable request data."""

    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{namespace}:{digest}"


class CacheManager:
    """Owns platform caches and provides uniform enable/disable behavior."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_size: int = 512,
        retrieval_ttl_seconds: int = 300,
        response_ttl_seconds: int = 300,
    ) -> None:
        self.enabled = enabled
        self.retrieval_cache: TTLCache[object] = TTLCache(
            max_size=max_size,
            ttl_seconds=retrieval_ttl_seconds,
        )
        self.response_cache: TTLCache[object] = TTLCache(
            max_size=max_size,
            ttl_seconds=response_ttl_seconds,
        )

    def stats(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "retrieval": self.retrieval_cache.stats(),
            "response": self.response_cache.stats(),
        }


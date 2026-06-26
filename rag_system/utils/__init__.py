"""Shared RAG utilities."""

from __future__ import annotations

import json


def debug_dump(title: str, payload: object, enabled: bool = False) -> None:
    """Print structured debug output when verbose mode is enabled."""
    if not enabled:
        return
    print(f"\n[{title}]")
    print(json.dumps(payload, indent=2, default=str))

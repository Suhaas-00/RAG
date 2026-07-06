"""Parent-child evidence aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceGroup:
    document_id: str
    source: str
    section: str
    chunks: tuple[dict[str, Any], ...]
    text: str


@dataclass(frozen=True)
class EvidenceBundle:
    groups: tuple[EvidenceGroup, ...] = ()
    context: str = ""


def build_parent_child_evidence(chunks: list[dict[str, Any]]) -> EvidenceBundle:
    """Merge adjacent child chunks and group evidence by document then section."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or id(chunk))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        document_id = str(chunk.get("document_id") or chunk.get("paper_id") or chunk.get("source") or "unknown")
        source = str(chunk.get("source") or "?")
        section = str(chunk.get("section") or "unknown")
        grouped[(document_id, source, section)].append(chunk)

    evidence_groups: list[EvidenceGroup] = []
    for (document_id, source, section), rows in grouped.items():
        rows.sort(key=lambda item: int(item.get("ordinal", item.get("page_number", 0)) or 0))
        text_parts: list[str] = []
        last_text = None
        for row in rows:
            text = str(row.get("text") or "").strip()
            if text and text != last_text:
                text_parts.append(text)
                last_text = text
        evidence_groups.append(
            EvidenceGroup(
                document_id=document_id,
                source=source,
                section=section,
                chunks=tuple(rows),
                text="\n".join(text_parts),
            )
        )

    context_parts: list[str] = []
    for group in evidence_groups:
        pages = sorted({str(c.get("page", c.get("page_number", "?"))) for c in group.chunks})
        confidence = max((float(c.get("confidence", c.get("final_score", 0.0)) or 0.0) for c in group.chunks), default=0.0)
        context_parts.append(
            f"[Source: {group.source} | Document: {group.document_id} | Section: {group.section} | "
            f"Pages: {', '.join(pages)} | Confidence: {confidence:.3f}]\n{group.text}"
        )
    return EvidenceBundle(groups=tuple(evidence_groups), context="\n\n".join(context_parts))

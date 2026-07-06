"""Central document registry and identifier index."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rag_system.retrieval.identifier_patterns import DEFAULT_IDENTIFIER_PATTERNS, normalize_identifier


SECTION_PRIORITY: dict[str, int] = {
    "abstract": 0,
    "introduction": 1,
    "background": 1,
    "methods": 2,
    "study design": 2,
    "results": 3,
    "discussion": 4,
    "conclusion": 5,
    "metadata": 6,
    "unknown": 99,
}


@dataclass(frozen=True)
class DocumentRecord:
    document_uuid: str
    document_id: str
    source: str
    paper_id: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    trial_ids: tuple[str, ...] = ()
    mesh: tuple[str, ...] = ()
    title: str | None = None
    aliases: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    publication: str | None = None
    year: str | None = None
    authors: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    chunk_map: dict[str, list[int]] = field(default_factory=dict)
    record_indices: tuple[int, ...] = ()


class DocumentRegistry:
    """Source of truth for indexed papers and O(1) identifier routing."""

    def __init__(self, documents: dict[str, DocumentRecord], identifier_index: dict[str, set[str]]) -> None:
        self.documents = documents
        self.identifier_index = identifier_index

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DocumentRegistry":
        records = payload.get("records", [])
        doc_metadata_index = payload.get("doc_metadata_index", {})
        paper_map = payload.get("paper_map", {})
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            doc_id = cls._document_key(record)
            groups[doc_id].append(idx)

        documents: dict[str, DocumentRecord] = {}
        identifier_index: dict[str, set[str]] = defaultdict(set)
        for doc_id, indices in groups.items():
            first = records[indices[0]]
            source = str(first.get("source") or first.get("filename") or doc_id)
            paper_id = first.get("paper_id") or cls._paper_id_for_source(source, paper_map)
            metadata = cls._metadata_for(source, paper_id, doc_metadata_index)
            document = cls._build_document(doc_id, source, paper_id, metadata, records, indices)
            documents[document.document_id] = document
            for alias in document.aliases:
                identifier_index[normalize_identifier(alias)].add(document.document_id)
            for identifier in cls._scan_identifiers(document, records, indices):
                identifier_index[normalize_identifier(identifier)].add(document.document_id)
        return cls(documents, dict(identifier_index))

    def resolve(self, identifier: str) -> set[str]:
        """Return document ids containing *identifier* in O(1)."""

        return set(self.identifier_index.get(normalize_identifier(identifier), set()))

    def get(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def allowed_indices(self, document_ids: Iterable[str]) -> set[int]:
        result: set[int] = set()
        for document_id in document_ids:
            doc = self.documents.get(document_id)
            if doc:
                result.update(doc.record_indices)
        return result

    def section_indices(self, document_ids: Iterable[str], section_hints: Iterable[str] = ()) -> list[int]:
        wanted = [section.casefold() for section in section_hints if section]
        indices: list[int] = []
        for document_id in document_ids:
            doc = self.documents.get(document_id)
            if not doc:
                continue
            if wanted:
                for section in wanted:
                    indices.extend(doc.chunk_map.get(section, []))
            else:
                for section in sorted(doc.chunk_map, key=lambda value: SECTION_PRIORITY.get(value, 50)):
                    indices.extend(doc.chunk_map[section])
        return list(dict.fromkeys(indices))

    @staticmethod
    def _document_key(record: dict[str, Any]) -> str:
        for key in ("document_id", "doc_id", "source_id", "paper_id", "source", "filename"):
            value = record.get(key)
            if value:
                return str(value)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key in ("pmid", "pmcid", "doi"):
            value = metadata.get(key)
            if value:
                return str(value)
        return str(record.get("chunk_id", "unknown-document"))

    @staticmethod
    def _paper_id_for_source(source: str, paper_map: dict[str, str]) -> str | None:
        for label, filename in paper_map.items():
            if Path(str(filename)).name.casefold() == Path(source).name.casefold():
                return label
        return None

    @staticmethod
    def _metadata_for(source: str, paper_id: str | None, metadata_index: dict[str, Any]) -> dict[str, Any]:
        keys = [source, Path(source).name, Path(source).stem, paper_id]
        for key in keys:
            if key and isinstance(metadata_index.get(key), dict):
                return dict(metadata_index[key])
        return {}

    @classmethod
    def _build_document(
        cls,
        doc_id: str,
        source: str,
        paper_id: str | None,
        metadata: dict[str, Any],
        records: list[dict[str, Any]],
        indices: list[int],
    ) -> DocumentRecord:
        sections: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            section = str(records[idx].get("section") or "unknown").casefold()
            sections[section].append(idx)
        trial_ids = tuple(sorted(cls._values(metadata, "trial_ids", "nct_ids", "clinical_trial_ids")))
        mesh = tuple(sorted(cls._values(metadata, "mesh", "mesh_terms")))
        entities = tuple(sorted(set(cls._values(metadata, "genes", "entities", "diseases"))))
        authors = tuple(cls._values(metadata, "authors"))
        aliases = {
            doc_id,
            source,
            Path(source).name,
            Path(source).stem,
            *(item for item in (paper_id, metadata.get("pmid"), metadata.get("pmcid"), metadata.get("doi")) if item),
            *trial_ids,
            *mesh,
        }
        document_uuid = hashlib.sha256(f"{doc_id}|{source}".encode("utf-8")).hexdigest()
        return DocumentRecord(
            document_uuid=document_uuid,
            document_id=doc_id,
            source=source,
            paper_id=str(paper_id) if paper_id else None,
            pmid=str(metadata.get("pmid")) if metadata.get("pmid") else None,
            pmcid=str(metadata.get("pmcid")) if metadata.get("pmcid") else None,
            doi=str(metadata.get("doi")) if metadata.get("doi") else None,
            trial_ids=trial_ids,
            mesh=mesh,
            title=str(metadata.get("title")) if metadata.get("title") else None,
            aliases=tuple(sorted(str(item) for item in aliases if item)),
            entities=entities,
            publication=str(metadata.get("journal") or metadata.get("publication") or "") or None,
            year=str(metadata.get("year")) if metadata.get("year") else None,
            authors=authors,
            sections=tuple(sorted(sections, key=lambda item: SECTION_PRIORITY.get(item, 50))),
            chunk_map={section: values for section, values in sections.items()},
            record_indices=tuple(indices),
        )

    @staticmethod
    def _values(metadata: dict[str, Any], *keys: str) -> set[str]:
        values: set[str] = set()
        for key in keys:
            raw = metadata.get(key)
            if not raw:
                continue
            if isinstance(raw, str):
                parts = re.split(r"[;,]", raw) if key.endswith("s") else [raw]
            else:
                parts = list(raw)
            values.update(str(item).strip() for item in parts if str(item).strip())
        return values

    @staticmethod
    def _scan_identifiers(document: DocumentRecord, records: list[dict[str, Any]], indices: list[int]) -> set[str]:
        identifiers = set(document.aliases)
        combined = "\n".join(str(records[idx].get("text", "")) for idx in indices)
        for pattern in DEFAULT_IDENTIFIER_PATTERNS:
            compiled = pattern.compile()
            for match in compiled.finditer(combined):
                raw = match.group(1) if match.groups() and match.group(1) else match.group(0)
                identifiers.add(raw)
        return identifiers

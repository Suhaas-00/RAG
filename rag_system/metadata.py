"""Paper metadata and chunk schema helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from rag_system.cleaner import canonical_section


DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s),;]+)", re.IGNORECASE)
PMID_RE = re.compile(r"\b(?:PMID|PubMed\s*ID)\s*[:#-]?\s*(\d{6,10})\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")

DISEASE_RE = re.compile(
    r"\b(?:cancer|carcinoma|adenocarcinoma|lymphoma|leukemia|melanoma|glioblastoma|"
    r"diabetes|hypertension|asthma|tuberculosis|pneumonia|sepsis|stroke|"
    r"alzheimer(?:'s)? disease|parkinson(?:'s)? disease|heart failure|"
    r"chronic kidney disease|non-small cell lung cancer|small cell lung cancer|nsclc)\b",
    re.IGNORECASE,
)
GENE_RE = re.compile(
    r"\b(?:EGFR|ALK|ROS1|KRAS|BRAF|MET|HER2|ERBB2|PIK3CA|PTEN|TP53|RET|"
    r"NTRK[123]|FGFR[1-4]|BRCA[12]|APC|VHL|T790M|L858R)\b",
    re.IGNORECASE,
)
STUDY_RE = re.compile(
    r"\b(?:randomi[sz]ed|double blind|placebo controlled|phase\s+[123I]{1,3}|"
    r"cohort|case-control|retrospective|prospective|clinical trial|meta-analysis|"
    r"systematic review|open-label)\b",
    re.IGNORECASE,
)


def build_paper_map(pdf_paths: Iterable[Path]) -> dict[str, str]:
    return {f"paper {i}": path.name for i, path in enumerate(sorted(pdf_paths), 1)}


def paper_id_for_source(source: str, paper_map: dict[str, str]) -> str:
    for label, filename in paper_map.items():
        if Path(filename).name.casefold() == Path(source).name.casefold():
            return label
    return Path(source).stem


def extract_document_metadata(text: str, source: str, paper_id: str) -> dict:
    head = text[:5000]
    doi = DOI_RE.search(head)
    pmid = PMID_RE.search(text)
    years = YEAR_RE.findall(head)
    diseases = sorted({m.group(0).lower() for m in DISEASE_RE.finditer(text)})
    genes = sorted({m.group(0).upper() for m in GENE_RE.finditer(text)})
    study_designs = sorted({m.group(0).lower() for m in STUDY_RE.finditer(text)})
    return {
        "source": source,
        "filename": Path(source).name,
        "paper_id": paper_id,
        "document_id": paper_id,
        "doi": doi.group(1).rstrip(".") if doi else None,
        "pmid": pmid.group(1) if pmid else None,
        "year": years[0] if years else None,
        "diseases": diseases,
        "genes": genes,
        "study_designs": study_designs,
    }


def normalize_record(record: dict, paper_id: str, chunk_id: str | None = None) -> dict:
    page = record.get("page", record.get("page_number"))
    normalized = dict(record)
    normalized["source"] = str(record.get("source", ""))
    normalized["filename"] = Path(normalized["source"]).name if normalized["source"] else ""
    normalized["paper_id"] = str(record.get("paper_id") or paper_id)
    normalized["document_id"] = str(record.get("document_id") or normalized["paper_id"])
    normalized["page"] = int(page or 1)
    normalized["page_number"] = normalized["page"]
    normalized["section"] = canonical_section(str(record.get("section") or "unknown"))
    normalized["chunk_id"] = chunk_id or str(record.get("chunk_id", ""))
    normalized["previous_chunk"] = record.get("previous_chunk", record.get("prev_chunk_id"))
    normalized["next_chunk"] = record.get("next_chunk", record.get("next_chunk_id"))
    normalized["prev_chunk_id"] = normalized["previous_chunk"]
    normalized["next_chunk_id"] = normalized["next_chunk"]
    normalized["token_count"] = int(record.get("token_count") or 0)
    normalized.setdefault("chunk_type", "content")
    normalized.setdefault("metadata", {})
    normalized["metadata"] = {
        **normalized["metadata"],
        "source": normalized["source"],
        "filename": normalized["filename"],
        "paper_id": normalized["paper_id"],
        "document_id": normalized["document_id"],
        "page": normalized["page"],
        "page_number": normalized["page_number"],
        "section": normalized["section"],
        "chunk_id": normalized["chunk_id"],
    }
    return normalized

"""Biomedical identifier extraction patterns.

The patterns are intentionally configurable at runtime by
``rag_system.retrieval.retrieval_config_yaml`` while keeping production-safe
defaults in code for environments that do not ship a YAML file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IdentifierPattern:
    """Compiled biomedical identifier pattern."""

    type: str
    pattern: str
    flags: int = re.IGNORECASE

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


DEFAULT_IDENTIFIER_PATTERNS: tuple[IdentifierPattern, ...] = (
    IdentifierPattern("nct", r"\bNCT\d{8}\b"),
    IdentifierPattern("pmid", r"\b(?:PMID[:\s]*)?(\d{6,10})\b"),
    IdentifierPattern("pmcid", r"\bPMC\d{4,10}\b"),
    IdentifierPattern("doi", r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"),
    IdentifierPattern("hgnc", r"\bHGNC:\d+\b"),
    IdentifierPattern("uniprot", r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])\b"),
    IdentifierPattern("gene_id", r"\b(?:GeneID|NCBI\s*Gene|Entrez)[:\s]*(\d{1,10})\b"),
    IdentifierPattern("mesh", r"\b(?:MeSH|D)[\s:]*([D]?\d{6})\b"),
    IdentifierPattern("umls", r"\bC\d{7}\b"),
    IdentifierPattern("drugbank", r"\bDB\d{5}\b"),
    IdentifierPattern("chebi", r"\bCHEBI:\d+\b"),
    IdentifierPattern("accession", r"\b(?:GSE|SRP|ERP|DRP|PRJNA|PRJEB|PRJDB|E-MTAB-)\d+\b"),
)


def normalize_identifier(value: str) -> str:
    """Canonicalize an identifier for O(1) dictionary lookup."""

    token = str(value or "").strip().strip(".,;:()[]{}")
    token = re.sub(r"^(PMID|PubMed ID|PMCID|DOI)\s*[:#]?\s*", "", token, flags=re.IGNORECASE)
    token = token.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return token.upper()


def compile_patterns(patterns: Iterable[IdentifierPattern] | None = None) -> dict[str, re.Pattern[str]]:
    """Compile identifier patterns keyed by identifier type."""

    return {item.type: item.compile() for item in (patterns or DEFAULT_IDENTIFIER_PATTERNS)}

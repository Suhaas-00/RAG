"""PDF text cleaning and biomedical section detection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


CANONICAL_SECTIONS = {
    "abstract": "abstract",
    "background": "background",
    "introduction": "introduction",
    "materials and methods": "methods",
    "material and methods": "methods",
    "patients and methods": "methods",
    "methods": "methods",
    "method": "methods",
    "study design": "study design",
    "experimental procedures": "methods",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
}

SECTION_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[\s.)-]*)?"
    r"(?P<section>abstract|background|introduction|materials?\s+and\s+methods|"
    r"patients\s+and\s+methods|methods?|study\s+design|experimental\s+procedures|"
    r"results?|findings|discussion|conclusions?)\s*[:.]?\s*$",
    re.IGNORECASE,
)

STOP_SECTION_RE = re.compile(
    r"^\s*(?:acknowledg(?:e)?ments?|conflicts?\s+of\s+interest|funding|"
    r"author\s+contributions?|references|bibliography|supplementary\s+material)\b",
    re.IGNORECASE,
)

FIGURE_TABLE_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?|table|supplementary\s+(?:fig(?:ure)?|table))\s*\d+",
    re.IGNORECASE,
)

BOILERPLATE_RE = re.compile(
    r"(downloaded\s+from|copyright|all\s+rights\s+reserved|creative\s+commons|"
    r"terms\s+of\s+use|published\s+by|licensee|correspondence\s+to|"
    r"received\s+\w+\s+\d|accepted\s+\w+\s+\d)",
    re.IGNORECASE,
)

AFFILIATION_RE = re.compile(
    r"\b(department|division|school|university|hospital|institute|center|centre|"
    r"laborator(?:y|ies)|faculty|college|clinic|foundation)\b",
    re.IGNORECASE,
)

AUTHOR_LINE_RE = re.compile(
    r"^(?:[A-Z][A-Za-z'`-]+(?:\s+[A-Z]\.?\s*){0,3})(?:,\s*| and\s+)"
)

SPACE_RE = re.compile(r"\s+")
SOFT_HYPHEN_RE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")


@dataclass(frozen=True)
class CleanPage:
    text: str
    page: int


def canonical_section(value: str | None) -> str:
    key = SPACE_RE.sub(" ", (value or "").strip().lower())
    return CANONICAL_SECTIONS.get(key, key or "unknown")


def detect_section_heading(line: str) -> str | None:
    match = SECTION_RE.match(line or "")
    if not match:
        return None
    return canonical_section(match.group("section"))


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SOFT_HYPHEN_RE.sub("", text)
    return text


def _line_quality(line: str, page: int) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    if STOP_SECTION_RE.match(stripped):
        return False
    if FIGURE_TABLE_RE.match(stripped):
        return False
    if BOILERPLATE_RE.search(stripped):
        return False
    if page <= 2 and AFFILIATION_RE.search(stripped) and len(stripped) < 180:
        return False
    if page <= 2 and AUTHOR_LINE_RE.match(stripped) and len(stripped.split()) < 35:
        return False
    digit_ratio = sum(ch.isdigit() for ch in stripped) / max(len(stripped), 1)
    if digit_ratio > 0.38:
        return False
    alpha_ratio = sum(ch.isalpha() for ch in stripped) / max(len(stripped), 1)
    return alpha_ratio >= 0.35


def remove_repeated_headers_footers(page_texts: list[str]) -> list[str]:
    line_counts: dict[str, int] = {}
    normalized_pages: list[list[str]] = []
    for text in page_texts:
        lines = [SPACE_RE.sub(" ", line.strip()) for line in normalize_text(text).splitlines()]
        normalized_pages.append(lines)
        candidates = lines[:3] + lines[-3:]
        for line in candidates:
            if 5 <= len(line) <= 120:
                line_counts[line.lower()] = line_counts.get(line.lower(), 0) + 1

    threshold = max(2, len(page_texts) // 3)
    repeated = {line for line, count in line_counts.items() if count >= threshold}
    cleaned_pages: list[str] = []
    for lines in normalized_pages:
        cleaned_pages.append("\n".join(line for line in lines if line.lower() not in repeated))
    return cleaned_pages


def clean_pdf_pages(pages: Iterable[object]) -> list[CleanPage]:
    """Remove common biomedical PDF noise while preserving section headings."""

    raw_pages: list[tuple[int, str]] = []
    for idx, page in enumerate(pages, 1):
        page_no = int(getattr(page, "page_number", getattr(page, "page", idx)))
        raw_pages.append((page_no, getattr(page, "text", "") or ""))

    without_headers = remove_repeated_headers_footers([text for _, text in raw_pages])
    cleaned: list[CleanPage] = []

    for (page_no, _), text in zip(raw_pages, without_headers):
        kept: list[str] = []
        skipping_tail = False
        for raw_line in normalize_text(text).splitlines():
            line = SPACE_RE.sub(" ", raw_line).strip()
            if not line:
                continue
            if STOP_SECTION_RE.match(line):
                skipping_tail = True
                continue
            heading = detect_section_heading(line)
            if heading:
                skipping_tail = False
                kept.append(heading.upper())
                continue
            if skipping_tail:
                continue
            if _line_quality(line, page_no):
                kept.append(line)
        cleaned.append(CleanPage(text="\n".join(kept), page=page_no))

    return cleaned


def clean_chunk_text(text: str | None) -> str:
    text = normalize_text(text)
    text = re.sub(r"\[(?:\d+[\s,;:-]*)+\]", " ", text)
    text = re.sub(r"\b(?:Fig|Figure|Table)\.?\s+\d+[^.]*\.", " ", text, flags=re.IGNORECASE)
    text = SPACE_RE.sub(" ", text).strip()
    return text


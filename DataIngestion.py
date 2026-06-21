"""
Medical Research PDF → FAISS VectorDB Pipeline
================================================
Industry-standard LangChain pipeline for processing medical research PDFs,
chunking semantically by section, enriching with metadata, embedding with
TF-IDF + SVD (fully offline, open-source), and storing in a FAISS vector DB.

Architecture:
    PDF Files
        └─► DirectoryLoader (LangChain)
                └─► PyPDFLoader per file
                        └─► SectionSplitter (custom LangChain TextSplitter)
                                └─► RecursiveCharacterTextSplitter (per section)
                                        └─► MetadataEnricher (entities, study type)
                                                └─► MedicalTFIDFEmbeddings (offline)
                                                        └─► FAISS VectorStore (LangChain)

Outputs:
    ./output/faiss_index/        — LangChain FAISS index (index.faiss + index.pkl)
    ./output/chunks_manifest.json — full chunk metadata manifest
    ./output/embedder.pkl        — fitted embedder (for query-time reuse)

Usage:
    python medical_vectordb_pipeline.py            # run full pipeline
    python medical_vectordb_pipeline.py --query "metformin HbA1c reduction"
"""

from __future__ import annotations
from functools import lru_cache
import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Any, Iterator, List, Optional


import numpy as np
from sklearn.pipeline import Pipeline as SklearnPipeline

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PDF_FOLDER    = "./pdfs"
OUTPUT_FOLDER = "./output"

# Chunking — characters (not tokens); RecursiveCharacterTextSplitter default unit
CHUNK_SIZE    = 1_500   # ≈ 300-400 tokens for medical English
CHUNK_OVERLAP = 200     # sentence-level overlap for context continuity

# Embedding
TFIDF_MAX_FEATURES = 50_000   # vocabulary ceiling
SVD_DIM            = 256      # latent semantic dimensions

# Retrieval defaults
DEFAULT_TOP_K = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

@lru_cache()
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. SECTION-AWARE TEXT SPLITTER  (custom LangChain TextSplitter)
# ─────────────────────────────────────────────────────────────────────────────
SECTION_HEADERS = {
    "abstract":     re.compile(r"^\s*(abstract)\s*$",                       re.I | re.M),
    "introduction": re.compile(r"^\s*(introduction|background)\s*$",        re.I | re.M),
    "methods":      re.compile(r"^\s*(methods?|materials?\s+and\s+methods?|methodology)\s*$", re.I | re.M),
    "results":      re.compile(r"^\s*(results?)\s*$",                       re.I | re.M),
    "discussion":   re.compile(r"^\s*(discussion)\s*$",                     re.I | re.M),
    "conclusion":   re.compile(r"^\s*(conclusions?|summary)\s*$",           re.I | re.M),
}

# Order matters — defines section priority in output
SECTION_ORDER = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]


class MedicalSectionSplitter(TextSplitter):
    """
    LangChain TextSplitter that:
    1. Detects IMRaD section boundaries in raw PDF text.
    2. Assigns each passage to its section.
    3. Delegates within-section chunking to RecursiveCharacterTextSplitter.
    4. Injects section, subtopic, prev_chunk_id, next_chunk_id into metadata.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._inner_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
        )

    # ── section parsing ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_sections(text: str) -> dict[str, str]:
        """Split raw text into {section_name: text} mapping."""
        sections: dict[str, str] = {s: "" for s in SECTION_ORDER}
        current = "abstract"
        for line in text.splitlines():
            detected = None
            for section, pat in SECTION_HEADERS.items():
                if pat.match(line.strip()):
                    detected = section
                    break
            if detected:
                current = detected
            else:
                sections[current] += line + "\n"
        return sections

    # ── chunk ID generation ───────────────────────────────────────────────────

    @staticmethod
    def _make_chunk_id(source: str, section: str, idx: int) -> str:
        key = f"{source}::{section}::{idx:05d}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    # ── LangChain TextSplitter interface ──────────────────────────────────────

    def split_text(self, text: str) -> List[str]:
        """Required by LangChain — returns plain text chunks (no metadata)."""
        sections = self._parse_sections(text)
        all_chunks: list[str] = []
        for section in SECTION_ORDER:
            body = sections.get(section, "").strip()
            if body:
                all_chunks.extend(self._inner_splitter.split_text(body))
        return all_chunks

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[Document]:
        """
        Override to produce richly-metadated Documents with:
        section, subtopic, chunk_id, prev_chunk_id, next_chunk_id, section_id.
        """
        all_docs: list[Document] = []

        for doc_idx, raw_text in enumerate(texts):
            base_meta = (metadatas or [{}])[doc_idx] if metadatas else {}
            source    = base_meta.get("source", f"doc_{doc_idx}")
            sections  = self._parse_sections(raw_text)

            for section in SECTION_ORDER:
                body = sections.get(section, "").strip()
                if not body:
                    continue

                section_docs = self._inner_splitter.create_documents(
                    [body], metadatas=[{**base_meta, "section": section}]
                )

                section_chunks: list[Document] = []
                for s_idx, doc in enumerate(section_docs):
                    chunk_id  = self._make_chunk_id(source, section, s_idx)
                    subtopic  = _infer_subtopic(doc.page_content)

                    doc.metadata.update({
                        "chunk_id":   chunk_id,
                        "section":    section,
                        "section_id": f"{source}::{section}",
                        "subtopic":   subtopic,
                        "source":     source,
                        # prev/next patched below after all chunks collected
                        "prev_chunk_id": None,
                        "next_chunk_id": None,
                    })
                    section_chunks.append(doc)

                # Wire prev/next within section
                for i, doc in enumerate(section_chunks):
                    doc.metadata["prev_chunk_id"] = (
                        section_chunks[i - 1].metadata["chunk_id"] if i > 0 else None
                    )
                    doc.metadata["next_chunk_id"] = (
                        section_chunks[i + 1].metadata["chunk_id"]
                        if i < len(section_chunks) - 1
                        else None
                    )

                all_docs.extend(section_chunks)

        return all_docs


# ─────────────────────────────────────────────────────────────────────────────
# 3. METADATA ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

# ── citation noise ────────────────────────────────────────────────────────────
_CITATION_RE = re.compile(
    r"\[\d+(?:[,–\-]\d+)*\]"                           # [1], [1,2], [1-3]
    r"|"
    r"\([A-Z][A-Za-z\s]+et al\.?,?\s*\d{4}[a-z]?\)"   # (Smith et al., 2020)
    r"|"
    r"\([A-Z][A-Za-z]+,?\s*\d{4}[a-z]?\)"              # (Smith, 2020)
)

# ── entity patterns ───────────────────────────────────────────────────────────
_ENTITY_PATTERNS: dict[str, re.Pattern] = {
    "disease": re.compile(
        r"\b(?:cancer|tumor|tumour|carcinoma|lymphoma|leukemia|melanoma|glioma|sarcoma"
        r"|diabetes(?:\s+mellitus)?|hypertension|stroke|myocardial\s+infarction"
        r"|COPD|asthma|fibrosis|hepatitis|cirrhosis|sepsis|pneumonia"
        r"|COVID-19|SARS-CoV-2|HIV|AIDS|tuberculosis"
        r"|dementia|Alzheimer(?:'s)?|Parkinson(?:'s)?|epilepsy"
        r"|schizophrenia|depression|anxiety|bipolar)\b", re.I
    ),
    "drug": re.compile(
        r"\b(?:aspirin|metformin|insulin|remdesivir|dexamethasone|vancomycin"
        r"|amoxicillin|ibuprofen|acetaminophen|warfarin|heparin|enoxaparin"
        r"|cisplatin|carboplatin|paclitaxel|docetaxel|oxaliplatin"
        r"|pembrolizumab|nivolumab|ipilimumab|atezolizumab"
        r"|trastuzumab|bevacizumab|rituximab|cetuximab"
        r"|tocilizumab|baricitinib|hydroxychloroquine|azithromycin"
        r"|prednisone|methylprednisolone|lisinopril|atorvastatin"
        r"|metoprolol|amlodipine|clopidogrel|rivaroxaban|apixaban)\b", re.I
    ),
    "procedure": re.compile(
        r"\b(?:surgery|biopsy|resection|transplant(?:ation)?"
        r"|radiotherapy|radiation\s+therapy|chemotherapy|immunotherapy|targeted\s+therapy"
        r"|intubation|mechanical\s+ventilation|extubation"
        r"|catheterisation|catheterization|angioplasty|bypass\s+surgery"
        r"|MRI|CT\s+scan|PET\s+scan|ultrasound|echocardiography"
        r"|colonoscopy|endoscopy|bronchoscopy|laparoscopy|laparotomy"
        r"|dialysis|haemodialysis|hemodialysis|plasmapheresis"
        r"|PCR|ELISA|sequencing|randomization|randomisation|blinding|masking)\b", re.I
    ),
    "outcome": re.compile(
        r"\b(?:overall\s+survival|progression-free\s+survival|disease-free\s+survival"
        r"|mortality|morbidity|survival\s+rate|remission|recurrence|relapse"
        r"|response\s+rate|complete\s+response|partial\s+response|stable\s+disease"
        r"|adverse\s+event|side\s+effect|toxicity|quality\s+of\s+life"
        r"|hazard\s+ratio|odds\s+ratio|relative\s+risk|confidence\s+interval"
        r"|p[- ]value|statistical\s+significance|efficacy|safety|tolerability)\b", re.I
    ),
}

# ── study type ────────────────────────────────────────────────────────────────
_STUDY_TYPES: list[tuple[str, re.Pattern]] = [
    ("randomized controlled trial",  re.compile(r"\b(?:randomized?\s+controlled\s+trial|RCT)\b", re.I)),
    ("meta-analysis",                re.compile(r"\bmeta[- ]analysis\b", re.I)),
    ("systematic review",            re.compile(r"\bsystematic\s+review\b", re.I)),
    ("cohort study",                 re.compile(r"\bcohort\s+study\b", re.I)),
    ("case-control study",           re.compile(r"\bcase[- ]control(?:\s+study)?\b", re.I)),
    ("cross-sectional study",        re.compile(r"\bcross[- ]sectional(?:\s+study)?\b", re.I)),
    ("observational study",          re.compile(r"\bobservational\s+study\b", re.I)),
    ("phase III trial",              re.compile(r"\bphase\s+III\s+(?:clinical\s+)?trial\b", re.I)),
    ("phase II trial",               re.compile(r"\bphase\s+II\s+(?:clinical\s+)?trial\b", re.I)),
    ("clinical trial",               re.compile(r"\bclinical\s+trial\b", re.I)),
    ("case report",                  re.compile(r"\bcase\s+report\b", re.I)),
]


def _clean_text(text: str) -> str:
    """Strip citation noise and normalise whitespace."""
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _extract_entities(text: str) -> List[str]:
    """Return list of 'type:value' entity strings found in text."""
    found: set[tuple[str, str]] = set()
    for etype, pat in _ENTITY_PATTERNS.items():
        for m in pat.finditer(text):
            found.add((etype, m.group().strip()))
    return sorted(f"{etype}:{val}" for etype, val in found)


def _detect_study_type(text: str) -> str:
    for label, pat in _STUDY_TYPES:
        if pat.search(text):
            return label
    return "unknown"


def _extract_year(text: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group() if m else "unknown"


def _infer_subtopic(text: str) -> str:
    """First sentence of the chunk, truncated to 120 chars."""
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return re.sub(r"[.!?]+$", "", first.strip())[:120]


def enrich_metadata(doc: Document, study_type: str, year: str) -> Document:
    """
    Attach entity, study_type, year, and cleaned text to a LangChain Document.
    Mutates in-place and returns the document.
    """
    cleaned = _clean_text(doc.page_content)
    doc.page_content = cleaned
    doc.metadata.update({
        "entities":    _extract_entities(cleaned),
        "study_type":  study_type,
        "year":        year,
        "db_source":   "PubMed",
        "chunk_type":  "semantic+structure",
    })
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# 4. TABLE HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_as_text(pdf_path: str) -> str:
    """
    Use pdfplumber to extract tables and return as structured JSON text.
    This text is prepended to the page text so LangChain loaders see it.
    """
    try:
        import pdfplumber
    except ImportError:
        return ""

    table_blocks: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for t_idx, table in enumerate(page.extract_tables() or []):
                    rows = [r for r in table if any(c for c in r if c)]
                    if len(rows) < 2:
                        continue
                    header   = [str(c or "").strip() for c in rows[0]]
                    data_rows = [[str(c or "").strip() for c in r] for r in rows[1:]]
                    tbl = {
                        "table_location": f"page_{page_num}_table_{t_idx}",
                        "headers": header,
                        "rows": data_rows,
                    }
                    table_blocks.append(f"\n[TABLE] {json.dumps(tbl, ensure_ascii=False)}\n")
    except Exception as e:
        log.warning("Table extraction failed for %s: %s", pdf_path, e)

    return "\n".join(table_blocks)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DOCUMENT LOADING  (LangChain DirectoryLoader + PyPDFLoader)
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_documents(pdf_folder: str) -> List[Document]:
    """
    Load all PDFs from a folder using LangChain's DirectoryLoader.
    Each page becomes a LangChain Document with source metadata.
    Tables are injected as structured JSON text blocks.
    """
    folder = Path(pdf_folder)
    if not folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    pdf_paths = list(folder.rglob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found in: {pdf_folder}")

    log.info("Found %d PDF file(s) in %s", len(pdf_paths), pdf_folder)

    # LangChain DirectoryLoader with PyPDFLoader per file
    loader = DirectoryLoader(
        str(folder),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,     # skip corrupt PDFs, log error
    )
    raw_docs: List[Document] = loader.load()
    log.info("Loaded %d raw page documents", len(raw_docs))

    # Inject table content
    table_cache: dict[str, str] = {}
    for doc in raw_docs:
        src = doc.metadata.get("source", "")
        if src not in table_cache:
            table_cache[src] = _extract_tables_as_text(src)
        if table_cache[src]:
            doc.page_content += table_cache[src]

    return raw_docs


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def _group_docs_by_source(docs: List[Document]) -> dict[str, List[Document]]:
    """Group page-level Documents by their source PDF path."""
    grouped: dict[str, List[Document]] = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        grouped.setdefault(src, []).append(doc)
    return grouped


def build_chunks(raw_docs: List[Document]) -> List[Document]:
    """
    Core LangChain chunking step:
    1. Group pages by source PDF.
    2. Merge pages into full-document text per PDF.
    3. Run MedicalSectionSplitter to produce section-tagged chunks.
    4. Enrich each chunk with entities, study type, year.
    """
    splitter   = MedicalSectionSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks: list[Document] = []

    grouped = _group_docs_by_source(raw_docs)
    log.info("Processing %d unique PDF source(s)", len(grouped))

    for source_path, pages in grouped.items():
        source_name = Path(source_path).stem
        # Merge all pages into one document text
        full_text   = "\n\n".join(p.page_content for p in pages)
        study_type  = _detect_study_type(full_text)
        year        = _extract_year(full_text)

        log.info(
            "  %-40s │ study_type=%-30s │ year=%s",
            source_name[:40], study_type, year
        )

        # LangChain create_documents: text → richly-metadated Document list
        chunks = splitter.create_documents(
            texts=[full_text],
            metadatas=[{"source": source_name}],
        )

        # Enrich each chunk
        for chunk in chunks:
            enrich_metadata(chunk, study_type, year)

        all_chunks.extend(chunks)
        log.info("    → %d chunks produced", len(chunks))

    return all_chunks


def build_vectorstore(chunks: List[Document], embedder: Embeddings):
    """
    Build LangChain FAISS vectorstore from chunks.
    Uses FAISS.from_documents which:
    - Calls embedder.embed_documents() on all chunk texts
    - Builds an IndexFlatIP (cosine similarity on normalised vectors)
    - Stores LangChain Document objects with full metadata in the docstore
    """
    log.info("Embedding %d chunks and building FAISS index...", len(chunks))
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedder,
    )
    log.info(
        "FAISS index built │ vectors=%d │ dim=%d",
        vectorstore.index.ntotal,
        vectorstore.index.d,
    )
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# 7. MANIFEST  (chunks_manifest.json)
# ─────────────────────────────────────────────────────────────────────────────

def save_manifest(chunks: List[Document], output_folder: str) -> None:
    """
    Save a full structured JSON manifest of all chunks with their metadata.
    Schema matches the Step 5 spec exactly.
    """
    records = []
    for chunk in chunks:
        m = chunk.metadata
        records.append({
            "chunk_id":      m.get("chunk_id", ""),
            "text":          chunk.page_content,
            "section":       m.get("section", ""),
            "subtopic":      m.get("subtopic", ""),
            "entities":      m.get("entities", []),
            "study_type":    m.get("study_type", "unknown"),
            "year":          m.get("year", "unknown"),
            "source":        m.get("db_source", "PubMed"),
            "chunk_type":    m.get("chunk_type", "semantic+structure"),
            "prev_chunk_id": m.get("prev_chunk_id"),
            "next_chunk_id": m.get("next_chunk_id"),
            "section_id":    m.get("section_id", ""),
        })

    manifest_path = Path(output_folder) / "chunks_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info("Manifest saved → %s (%d records)", manifest_path, len(records))


# ─────────────────────────────────────────────────────────────────────────────
# 8. RETRIEVAL  (LangChain similarity search)
# ─────────────────────────────────────────────────────────────────────────────

class MedicalVectorDB:
    """
    High-level retrieval interface wrapping LangChain FAISS vectorstore.

    Supports:
    - similarity_search: plain top-k retrieval
    - similarity_search_with_score: retrieval + cosine similarity scores
    - section_filtered_search: restrict results to a specific IMRaD section
    - mmr_search: Maximal Marginal Relevance for diverse results
    """

    def __init__(self, vectorstore: FAISS, embedder: Embeddings):
        self.vectorstore = vectorstore
        self.embedder    = embedder

    # ── core retrieval ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        section: Optional[str] = None,
    ) -> List[dict]:
        """
        Retrieve top-k chunks for a query.
        Optionally filter by IMRaD section.
        Returns list of structured result dicts.
        """
        # Over-fetch for post-filtering
        fetch_k = top_k * 4 if section else top_k

        results_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=fetch_k,
        )

        output = []
        for doc, score in results_with_scores:
            if section and doc.metadata.get("section") != section:
                continue
            output.append({
                "score":         round(float(score), 4),
                "chunk_id":      doc.metadata.get("chunk_id"),
                "section":       doc.metadata.get("section"),
                "subtopic":      doc.metadata.get("subtopic"),
                "entities":      doc.metadata.get("entities", []),
                "study_type":    doc.metadata.get("study_type"),
                "year":          doc.metadata.get("year"),
                "prev_chunk_id": doc.metadata.get("prev_chunk_id"),
                "next_chunk_id": doc.metadata.get("next_chunk_id"),
                "text":          doc.page_content,
            })
            if len(output) >= top_k:
                break

        return output

    def mmr_search(self, query: str, top_k: int = DEFAULT_TOP_K, fetch_k: int = 20) -> List[dict]:
        """
        Maximal Marginal Relevance search — balances relevance with diversity.
        Useful when chunks are very similar (e.g. repeated methodology paragraphs).
        """
        docs = self.vectorstore.max_marginal_relevance_search(
            query=query, k=top_k, fetch_k=fetch_k
        )
        return [
            {
                "chunk_id": d.metadata.get("chunk_id"),
                "section":  d.metadata.get("section"),
                "entities": d.metadata.get("entities", []),
                "text":     d.page_content,
            }
            for d in docs
        ]

    # ── as LangChain retriever ────────────────────────────────────────────────

    def as_retriever(self, top_k: int = DEFAULT_TOP_K):
        """Return a LangChain BaseRetriever for use in chains/agents."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, output_folder: str) -> None:
        index_path = str(Path(output_folder) / "faiss_index")
        self.vectorstore.save_local(index_path)
        log.info("FAISS vectorstore saved → %s/", index_path)

    @classmethod
    def load(cls, output_folder: str) -> "MedicalVectorDB":
        embedder_path = str(Path(output_folder) / "embedder.pkl")
        index_path    = str(Path(output_folder) / "faiss_index")
        embedder = get_embedder()
        vectorstore = FAISS.load_local(
            index_path,
            embedder,
            allow_dangerous_deserialization=True,
        )
        log.info(
            "VectorDB loaded │ vectors=%d │ dim=%d",
            vectorstore.index.ntotal,
            vectorstore.index.d,
        )
        return cls(vectorstore=vectorstore, embedder=embedder)


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> MedicalVectorDB:
    """
    Full pipeline:  PDF folder → FAISS VectorDB + manifest

    Steps:
    1. Load PDFs with LangChain DirectoryLoader
    2. Chunk + section-tag with MedicalSectionSplitter
    3. Embed with MedicalTFIDFEmbeddings
    4. Build LangChain FAISS vectorstore
    5. Save index, embedder, manifest
    """
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info("  MEDICAL PDF → FAISS VECTORDB PIPELINE")
    log.info("═" * 60)

    # Step 1 — Load
    raw_docs = load_pdf_documents(PDF_FOLDER)

    # Step 2 — Chunk
    chunks = build_chunks(raw_docs)
    log.info("Total chunks: %d", len(chunks))

    if not chunks:
        log.error("No chunks produced — check PDF content and folder path.")
        return None

    # Step 3+4 — Embed + Index
    embedder = get_embedder()
    vectorstore = build_vectorstore(chunks, embedder)
    db          = MedicalVectorDB(vectorstore, embedder)

    # Step 5 — Persist
    db.save(OUTPUT_FOLDER)
    save_manifest(chunks, OUTPUT_FOLDER)

    log.info("═" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("  Index  → %s/faiss_index/", OUTPUT_FOLDER)
    log.info("  Chunks → %s/chunks_manifest.json", OUTPUT_FOLDER)
    log.info("═" * 60)

    return db




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical PDF → FAISS VectorDB Pipeline")
    parser.add_argument("--query",   type=str, default=None, help="Query the existing index")
    parser.add_argument("--top_k",  type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--section", type=str, default=None,
        choices=["abstract", "introduction", "methods", "results", "discussion", "conclusion"],
    )
    args = parser.parse_args()
    run_pipeline()
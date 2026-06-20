"""
Medical Research PDF → FAISS VectorDB Pipeline
================================================
Processes all PDFs in ./pdf/ folder, chunks them semantically per section,
enriches with metadata, embeds with SentenceTransformers, and stores in FAISS.

Usage:
    python medical_vectordb_pipeline.py

Output:
    ./output/chunks.json          — all structured chunk metadata
    ./output/faiss_index.bin      — FAISS index (L2 / flat)
    ./output/chunk_id_map.json    — maps FAISS row → chunk_id for retrieval
"""

import os
import re
import json
import glob
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pdfplumber
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PDF_FOLDER   = "./pdfs"
OUTPUT_FOLDER = "./output"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, fast, good for scientific text
CHUNK_MIN_TOKENS = 300
CHUNK_MAX_TOKENS = 500
FAISS_DIM    = 384                    # must match EMBED_MODEL output dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TOKENIZER  (offline NLTK word tokenizer)
# Approximation: 1 word ≈ 1.3 tokens (common for medical text)
# ─────────────────────────────────────────────
nltk.download("punkt_tab", quiet=True)

def count_tokens(text: str) -> int:
    """Offline token estimator: NLTK word count × 1.3 (medical text skews longer)."""
    from nltk.tokenize import word_tokenize
    return int(len(word_tokenize(text)) * 1.3)


# ─────────────────────────────────────────────
# SECTION DETECTION
# ─────────────────────────────────────────────
SECTION_PATTERNS = {
    "abstract":     re.compile(r"^\s*(abstract)\b",                   re.I),
    "introduction": re.compile(r"^\s*(introduction|background)\b",    re.I),
    "methods":      re.compile(r"^\s*(methods?|materials?\s+and\s+methods?|methodology)\b", re.I),
    "results":      re.compile(r"^\s*(results?)\b",                   re.I),
    "discussion":   re.compile(r"^\s*(discussion)\b",                 re.I),
    "conclusion":   re.compile(r"^\s*(conclusions?|summary)\b",       re.I),
}

def detect_section(line: str) -> Optional[str]:
    """Return section name if line is a section header, else None."""
    stripped = line.strip()
    # Must be short (headings are short) and not a full sentence
    if len(stripped) > 80 or stripped.endswith("."):
        return None
    for section, pat in SECTION_PATTERNS.items():
        if pat.match(stripped):
            return section
    return None


# ─────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text page by page.
    Returns list of { page_num, text }.
    Tables are captured as descriptive JSON-like text.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Extract tables first and replace them with structured text
            tables = page.extract_tables()
            table_texts = []
            for t_idx, table in enumerate(tables):
                rows = [row for row in table if any(c for c in row)]
                if not rows:
                    continue
                header = rows[0]
                data_rows = rows[1:]
                tbl_json = {
                    "table_index": t_idx,
                    "headers": header,
                    "rows": data_rows
                }
                table_texts.append(
                    f"[TABLE {t_idx}] " + json.dumps(tbl_json, ensure_ascii=False)
                )

            raw_text = page.extract_text() or ""
            combined = raw_text + ("\n" + "\n".join(table_texts) if table_texts else "")
            pages.append({"page_num": i, "text": combined})
    return pages


# ─────────────────────────────────────────────
# CITATION NOISE REMOVAL
# ─────────────────────────────────────────────
_citation_re = re.compile(
    r"\[\d+(?:[,\-]\d+)*\]"             # [1], [1,2], [1-3]
    r"|"
    r"\([A-Za-z][A-Za-z\s]+et al\.,?\s*\d{4}\)"  # (Smith et al., 2020)
    r"|"
    r"\([A-Za-z][A-Za-z\s]+,\s*\d{4}\)"          # (Smith, 2020)
)

def clean_text(text: str) -> str:
    text = _citation_re.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────
_sent_split_re = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
)

def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter safe for medical text (avoids splitting 'e.g.', abbreviations)."""
    parts = _sent_split_re.split(text)
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────
# SEMANTIC ENTITY EXTRACTION  (regex-based, no spaCy dependency)
# ─────────────────────────────────────────────
DISEASE_KW   = re.compile(r"\b(?:cancer|tumor|tumour|diabetes|hypertension|stroke|infarction|carcinoma|lymphoma|leukemia|sepsis|pneumonia|COVID-19|SARS-CoV-2|asthma|COPD|fibrosis|melanoma|glioma|hepatitis|cirrhosis|dementia|Alzheimer|Parkinson|epilepsy|schizophrenia|depression|anxiety)\b", re.I)
DRUG_KW      = re.compile(r"\b(?:aspirin|metformin|insulin|remdesivir|dexamethasone|vancomycin|amoxicillin|ibuprofen|acetaminophen|warfarin|heparin|cisplatin|carboplatin|paclitaxel|docetaxel|pembrolizumab|nivolumab|trastuzumab|bevacizumab|rituximab|tocilizumab|hydroxychloroquine|azithromycin|prednisone|lisinopril|atorvastatin|metoprolol|amlodipine)\b", re.I)
PROCEDURE_KW = re.compile(r"\b(?:surgery|biopsy|resection|transplant|radiotherapy|chemotherapy|immunotherapy|intubation|ventilation|catheterization|angioplasty|bypass|MRI|CT scan|PET scan|ultrasound|echocardiography|colonoscopy|endoscopy|laparoscopy|dialysis|plasmapheresis|ELISA|PCR|sequencing|randomization|blinding)\b", re.I)
OUTCOME_KW   = re.compile(r"\b(?:mortality|survival|remission|recurrence|progression|response rate|adverse event|side effect|quality of life|efficacy|safety|hazard ratio|odds ratio|confidence interval|p-value|median survival|overall survival|progression-free survival|disease-free survival)\b", re.I)

def extract_entities(text: str) -> list[str]:
    entities = set()
    for m in DISEASE_KW.finditer(text):    entities.add(("disease", m.group()))
    for m in DRUG_KW.finditer(text):       entities.add(("drug", m.group()))
    for m in PROCEDURE_KW.finditer(text):  entities.add(("procedure", m.group()))
    for m in OUTCOME_KW.finditer(text):    entities.add(("outcome", m.group()))
    return [f"{etype}:{val}" for etype, val in sorted(entities)]


# ─────────────────────────────────────────────
# STUDY TYPE DETECTION
# ─────────────────────────────────────────────
STUDY_TYPES = {
    "randomized controlled trial": re.compile(r"\brandomized\s+controlled\s+trial|RCT\b", re.I),
    "meta-analysis":               re.compile(r"\bmeta.analysis\b", re.I),
    "systematic review":           re.compile(r"\bsystematic\s+review\b", re.I),
    "cohort study":                re.compile(r"\bcohort\s+study\b", re.I),
    "case-control study":          re.compile(r"\bcase.control\b", re.I),
    "cross-sectional study":       re.compile(r"\bcross.sectional\b", re.I),
    "observational study":         re.compile(r"\bobservational\s+study\b", re.I),
    "case report":                 re.compile(r"\bcase\s+report\b", re.I),
    "clinical trial":              re.compile(r"\bclinical\s+trial\b", re.I),
}

def detect_study_type(full_text: str) -> str:
    for label, pat in STUDY_TYPES.items():
        if pat.search(full_text):
            return label
    return "unknown"


# ─────────────────────────────────────────────
# SUBTOPIC INFERENCE
# ─────────────────────────────────────────────
def infer_subtopic(section: str, text: str) -> str:
    """Generate a brief subtopic label from the first meaningful sentence."""
    sentences = split_sentences(text)
    if not sentences:
        return section
    first = sentences[0][:120].strip()
    # Capitalise, remove trailing punctuation
    return re.sub(r"[.!?]+$", "", first)


# ─────────────────────────────────────────────
# YEAR EXTRACTION
# ─────────────────────────────────────────────
_year_re = re.compile(r"\b(19|20)\d{2}\b")

def extract_year(text: str) -> str:
    m = _year_re.search(text)
    return m.group() if m else "unknown"


# ─────────────────────────────────────────────
# CORE CHUNKING
# ─────────────────────────────────────────────
def chunk_id(source: str, section: str, idx: int) -> str:
    base = f"{source}_{section}_{idx:04d}"
    return hashlib.md5(base.encode()).hexdigest()[:12]


def build_chunks(sections: dict[str, str], source: str, study_type: str, year: str) -> list[dict]:
    """
    For each section, split into sentences, merge into token-bounded chunks,
    and produce metadata dicts.
    """
    all_chunks: list[dict] = []

    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue

        sentences = split_sentences(clean_text(section_text))
        current_sentences: list[str] = []
        current_tokens = 0
        section_chunks: list[dict] = []
        section_idx = 0

        def flush_chunk():
            nonlocal current_sentences, current_tokens, section_idx
            if not current_sentences:
                return
            text_block = " ".join(current_sentences)
            cid = chunk_id(source, section_name, section_idx)
            section_chunks.append({
                "_cid": cid,
                "text": text_block,
                "section": section_name,
                "entities": extract_entities(text_block),
                "subtopic": infer_subtopic(section_name, text_block),
            })
            section_idx += 1
            current_sentences = []
            current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens(sent)

            # Single sentence exceeds max — emit alone
            if sent_tokens > CHUNK_MAX_TOKENS:
                flush_chunk()
                cid = chunk_id(source, section_name, section_idx)
                section_chunks.append({
                    "_cid": cid,
                    "text": sent,
                    "section": section_name,
                    "entities": extract_entities(sent),
                    "subtopic": infer_subtopic(section_name, sent),
                })
                section_idx += 1
                continue

            # Would overflow max — flush first
            if current_tokens + sent_tokens > CHUNK_MAX_TOKENS:
                # Only flush if we've hit minimum
                if current_tokens >= CHUNK_MIN_TOKENS:
                    flush_chunk()
                else:
                    # Keep adding even if a bit over min to avoid tiny chunks
                    pass

            current_sentences.append(sent)
            current_tokens += sent_tokens

            # Natural flush when we hit the sweet spot
            if current_tokens >= CHUNK_MIN_TOKENS:
                flush_chunk()

        flush_chunk()  # Remainder

        # Assign prev/next IDs within section
        for i, chunk in enumerate(section_chunks):
            cid = chunk.pop("_cid")
            prev_id = section_chunks[i - 1].get("chunk_id", None) if i > 0 else None
            # next_id will be patched in the next iteration
            chunk["chunk_id"] = cid
            chunk["prev_chunk_id"] = prev_id
            chunk["next_chunk_id"] = None  # filled below
            chunk["section_id"] = f"{source}_{section_name}"
            chunk["study_type"] = study_type
            chunk["year"] = year
            chunk["source"] = "PubMed"
            chunk["chunk_type"] = "semantic+structure"

        # Patch next_chunk_id
        for i in range(len(section_chunks) - 1):
            section_chunks[i]["next_chunk_id"] = section_chunks[i + 1]["chunk_id"]

        all_chunks.extend(section_chunks)

    return all_chunks


# ─────────────────────────────────────────────
# SECTION ASSEMBLY FROM RAW PAGES
# ─────────────────────────────────────────────
def assemble_sections(pages: list[dict]) -> dict[str, str]:
    """Walk pages line-by-line, assign text to detected sections."""
    sections: dict[str, str] = {
        "abstract": "", "introduction": "", "methods": "",
        "results": "", "discussion": "", "conclusion": ""
    }
    current_section = "abstract"  # Default before any heading found

    for page in pages:
        for line in page["text"].splitlines():
            detected = detect_section(line)
            if detected:
                current_section = detected
            else:
                sections[current_section] += line + "\n"

    return sections


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def process_pdf(pdf_path: str) -> list[dict]:
    source = Path(pdf_path).stem
    log.info(f"Processing: {pdf_path}")

    pages = extract_text_from_pdf(pdf_path)
    full_text = "\n".join(p["text"] for p in pages)

    study_type = detect_study_type(full_text)
    year = extract_year(full_text)

    sections = assemble_sections(pages)
    chunks = build_chunks(sections, source, study_type, year)

    log.info(f"  → {len(chunks)} chunks | study_type={study_type} | year={year}")
    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    # IndexFlatIP = cosine similarity (since embeddings are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def run_pipeline():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "**", "*.pdf"), recursive=True)
    if not pdf_files:
        log.warning(f"No PDFs found in '{PDF_FOLDER}'. Place PDFs there and re-run.")
        return

    log.info(f"Found {len(pdf_files)} PDF(s)")

    all_chunks: list[dict] = []
    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        try:
            chunks = process_pdf(pdf_path)
            all_chunks.extend(chunks)
        except Exception as e:
            log.error(f"Failed to process {pdf_path}: {e}", exc_info=True)

    if not all_chunks:
        log.error("No chunks produced. Aborting.")
        return

    log.info(f"Total chunks across all PDFs: {len(all_chunks)}")

    # Save chunks JSON
    chunks_path = os.path.join(OUTPUT_FOLDER, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    log.info(f"Chunks saved → {chunks_path}")

    # Embed
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_chunks(all_chunks, model)

    # Build + save FAISS index
    index = build_faiss_index(embeddings)
    index_path = os.path.join(OUTPUT_FOLDER, "faiss_index.bin")
    faiss.write_index(index, index_path)
    log.info(f"FAISS index saved → {index_path}")

    # Save chunk_id map (FAISS row → chunk_id)
    id_map = {i: c["chunk_id"] for i, c in enumerate(all_chunks)}
    map_path = os.path.join(OUTPUT_FOLDER, "chunk_id_map.json")
    with open(map_path, "w") as f:
        json.dump(id_map, f, indent=2)
    log.info(f"Chunk ID map saved → {map_path}")

    log.info("✅ Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
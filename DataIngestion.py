"""Build synchronised FAISS and lexical indexes from a directory of PDFs.

Pipeline
--------
1.  Discover all ``*.pdf`` files under *pdf_dir* (recursive).
2.  Extract page text via ``pypdf``.
3.  Extract document-level metadata heuristically (DOI, PMID, year, journal,
    diseases, genes, study designs).
4.  Produce **content chunks** (sentence/section-aware) and **metadata chunks**
    (one chunk per structured field + abstract) using the dual-strategy chunker.
5.  Filter weak chunks (too short, too noisy, or non-alphabetic) — but always
    keep metadata chunks, which are short by design.
6.  Wire doubly-linked chunk IDs on the surviving content chunks.
7.  Encode all chunks with :class:`PubMedEmbedder` (chunk-type-prefix aware).
8.  Build a FAISS ``IndexFlatIP`` over the unit-normalised vectors.
9.  Build a TF-IDF ``(1,2)``-gram matrix for keyword re-scoring at query time.
10. Serialise FAISS vectors, a metadata pickle, and a human-readable JSON
    manifest to *output_dir*.

The payload pickle is the authoritative artefact consumed by the retriever.
It carries:
- ``records``                – all kept chunk dicts (content + metadata)
- ``id_to_position``         – chunk_id → FAISS row index
- ``tfidf_vectorizer``       – fitted sklearn TfidfVectorizer
- ``tfidf_matrix``           – scipy sparse matrix (n_chunks × n_features)
- ``doc_metadata_index``     – source_stem → extracted metadata dict
- ``schema_version``         – int version tag for forward-compatibility checks

Public API
----------
load_pdf(path)                    – Extract page texts from a single PDF.
ingest(pdf_dir, output_dir, ...)  – Full pipeline; returns chunk count.
main()                            – CLI entry point.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_system.ingestion.chunking import PageText, chunk_pages, wire_neighbors
from rag_system.ingestion.embedding import PubMedEmbedder
from rag_system.metadata import build_paper_map, extract_document_metadata, normalize_record
from rag_system.utils.config import Settings
from rag_system.utils.preprocessing import (
    normalize_for_embedding,
    token_count,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum ratio of alphabetic characters for a **content** chunk.
_MIN_ALPHA_RATIO: float = 0.45

#: Payload schema version — bump when the pickle format changes.
_SCHEMA_VERSION: int = 3

#: Preprocessing tag embedded in the payload for compatibility checks.
_PREPROCESSING_TAG: str = "normalize_for_embedding:v1"


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------


def load_pdf(path: Path) -> list[PageText]:
    """Extract per-page text from *path* using ``pypdf``.

    Parameters
    ----------
    path:
        Absolute or relative path to a ``.pdf`` file.

    Returns
    -------
    List of :class:`~rag_system.ingestion.chunking.PageText` objects, one per
    page (1-based page numbers).  Pages that yield no text are included as
    empty-string entries so that page provenance stays accurate.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be opened as a PDF.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    log.debug("Reading PDF: %s", path)
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF '{path}': {exc}") from exc

    pages = [
        PageText(page.extract_text() or "", page_number=number)
        for number, page in enumerate(reader.pages, 1)
    ]
    log.debug("  → %d page(s) extracted from %s", len(pages), path.name)
    return pages


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def _is_quality_chunk(chunk: dict, min_tokens: int) -> bool:
    """Return ``True`` when the chunk meets minimum quality criteria.

    Metadata chunks always pass so that structured field information is never
    silently dropped.

    Criteria for **content** chunks
    --------------------------------
    1. Token count ≥ *min_tokens*.
    2. Alphabetic character ratio ≥ :data:`_MIN_ALPHA_RATIO` (filters OCR
       tables, header/footer lines, numeric-heavy boilerplate).
    """
    # Metadata chunks are kept unconditionally — they are short by design.
    if chunk.get("chunk_type") == "metadata":
        return True

    text = chunk.get("text", "")
    if token_count(text) < min_tokens:
        return False
    length = max(len(text), 1)
    if sum(ch.isalpha() for ch in text) / length < _MIN_ALPHA_RATIO:
        return False
    return True


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------


def ingest(
    pdf_dir: str | Path,
    output_dir: str | Path,
    settings: Optional[Settings] = None,
) -> int:
    """Run the full ingestion pipeline and write the index to *output_dir*.

    Parameters
    ----------
    pdf_dir:
        Directory (searched recursively) containing source ``.pdf`` files.
    output_dir:
        Root output directory.  The FAISS index is written under
        ``output_dir/faiss_index/``; a human-readable JSON manifest is
        written to ``output_dir/chunks_manifest.json``.
    settings:
        Optional :class:`~rag_system.utils.config.Settings` instance.
        When ``None``, a default instance is constructed from *pdf_dir* and
        *output_dir*.

    Returns
    -------
    Total number of chunks written to the index (content + metadata).

    Raises
    ------
    ValueError
        If no PDFs are found, or if no usable chunks are produced.
    FileNotFoundError
        If *pdf_dir* does not exist.
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"pdf_dir does not exist: {pdf_dir}")

    settings = settings or Settings(pdf_dir=pdf_dir, output_dir=output_dir)

    t0 = time.perf_counter()
    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found under '{pdf_dir}'")

    log.info("Discovered %d PDF file(s) under '%s'", len(pdf_paths), pdf_dir)

    # -----------------------------------------------------------------------
    # Stage 1 – Extract, chunk, filter
    # -----------------------------------------------------------------------
    all_records: list[dict] = []
    per_pdf_stats: list[dict] = []
    doc_metadata_index: dict[str, dict] = {}
    paper_map = build_paper_map(pdf_paths)
    source_to_paper = {filename: label for label, filename in paper_map.items()}

    for pdf_path in pdf_paths:
        source = pdf_path.name
        paper_id = source_to_paper.get(source, pdf_path.stem)
        source_stem = pdf_path.stem
        try:
            pages = load_pdf(pdf_path)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Skipping '%s': %s", source, exc)
            continue

        # ── Document-level metadata extraction ─────────────────────────────
        full_text = "\n".join(p.text for p in sorted(pages, key=lambda p: p.page_number))
        doc_meta = extract_document_metadata(full_text, source=source, paper_id=paper_id)
        doc_metadata_index[source] = doc_meta
        doc_metadata_index[source_stem] = doc_meta
        doc_metadata_index[paper_id] = doc_meta
        log.info(
            "  %s: diseases=%s  genes=%s  year=%s",
            source,
            doc_meta.get("diseases", []),
            doc_meta.get("genes", []),
            doc_meta.get("year"),
        )

        # ── Dual-strategy chunking (content + metadata) ─────────────────────
        candidates = chunk_pages(
            pages,
            source=source,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            doc_metadata=doc_meta,
        )
        candidates = [
            normalize_record(candidate, paper_id=paper_id, chunk_id=candidate.get("chunk_id"))
            for candidate in candidates
        ]

        # ── Quality filtering (metadata chunks always pass) ─────────────────
        kept = [c for c in candidates if _is_quality_chunk(c, settings.min_chunk_tokens)]

        # ── Wire neighbour IDs on content chunks only ───────────────────────
        wire_neighbors(kept)
        for chunk in kept:
            chunk["previous_chunk"] = chunk.get("prev_chunk_id")
            chunk["next_chunk"] = chunk.get("next_chunk_id")

        all_records.extend(kept)

        n_content = sum(1 for c in kept if c.get("chunk_type") != "metadata")
        n_meta = sum(1 for c in kept if c.get("chunk_type") == "metadata")
        per_pdf_stats.append(
            {
                "source":         source,
                "pages":          len(pages),
                "raw_chunks":     len(candidates),
                "kept_content":   n_content,
                "kept_metadata":  n_meta,
                "kept_total":     len(kept),
                "doc_metadata":   doc_meta,
            }
        )
        log.info(
            "  %s: %d pages → %d raw → %d content + %d metadata chunks kept",
            source, len(pages), len(candidates), n_content, n_meta,
        )

    if not all_records:
        raise ValueError(
            "PDFs were read but no usable chunks were produced. "
            "Check that PDFs contain extractable text and that min_chunk_tokens is not too high."
        )

    log.info(
        "Total chunks after filtering: %d  (%d content, %d metadata)",
        len(all_records),
        sum(1 for c in all_records if c.get("chunk_type") != "metadata"),
        sum(1 for c in all_records if c.get("chunk_type") == "metadata"),
    )

    # -----------------------------------------------------------------------
    # Stage 2 – Encode with PubMedEmbedder (chunk-type aware)
    # -----------------------------------------------------------------------
    embedder = PubMedEmbedder(settings.model_name, use_prefixes=True)

    log.info("Encoding %d chunks (batch_size=%d)…", len(all_records), settings.embedding_batch_size)
    vectors: np.ndarray = embedder.encode_chunks(
        all_records,
        batch_size=settings.embedding_batch_size,
    )

    # -----------------------------------------------------------------------
    # Stage 3 – Build FAISS index
    # -----------------------------------------------------------------------
    index = faiss.IndexFlatIP(embedder.dimension)
    index.add(vectors)
    log.info("FAISS IndexFlatIP ready — ntotal=%d, dim=%d", index.ntotal, index.d)

    # -----------------------------------------------------------------------
    # Stage 4 – Build TF-IDF matrix
    # -----------------------------------------------------------------------
    log.info("Fitting TF-IDF vectoriser…")
    normalised_texts = [normalize_for_embedding(c["text"]) for c in all_records]
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        analyzer="word",
    )
    keyword_matrix = tfidf.fit_transform(normalised_texts)
    log.info(
        "TF-IDF matrix: %d documents × %d features",
        keyword_matrix.shape[0],
        keyword_matrix.shape[1],
    )

    # -----------------------------------------------------------------------
    # Stage 5 – Serialise outputs
    # -----------------------------------------------------------------------
    index_dir = output_dir / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # 5a — FAISS vectors
    vectors_path = index_dir / "vectors.index"
    faiss.write_index(index, str(vectors_path))
    log.info("FAISS index written to '%s'", vectors_path)

    # 5b — Metadata pickle (the authoritative retrieval payload)
    payload: dict = {
        "schema_version":      _SCHEMA_VERSION,
        "model_name":          settings.model_name,
        "embedding_dimension": embedder.dimension,
        "preprocessing":       _PREPROCESSING_TAG,
        "records":             all_records,
        "id_to_position":      {item["chunk_id"]: i for i, item in enumerate(all_records)},
        "tfidf_vectorizer":    tfidf,
        "tfidf_matrix":        keyword_matrix,
        "doc_metadata_index":  doc_metadata_index,
        "paper_map":           paper_map,
        "per_pdf_stats":       per_pdf_stats,
        "settings":            settings.to_dict(),
    }
    metadata_path = index_dir / "metadata.pkl"
    with metadata_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Metadata pickle written to '%s'", metadata_path)

    # 5c — Human-readable JSON manifest
    manifest_path = output_dir / "chunks_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(all_records, fh, ensure_ascii=False, indent=2, default=str)
    log.info("Chunk manifest written to '%s'", manifest_path)

    # 5d — Doc metadata index (separate JSON for easy inspection)
    doc_meta_path = output_dir / "doc_metadata_index.json"
    with doc_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(doc_metadata_index, fh, ensure_ascii=False, indent=2)
    log.info("Doc metadata index written to '%s'", doc_meta_path)

    elapsed = time.perf_counter() - t0
    log.info(
        "Ingestion complete — %d chunks indexed in %.1fs (%.0f chunks/s)",
        len(all_records),
        elapsed,
        len(all_records) / max(elapsed, 1e-6),
    )
    return len(all_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DataIngestion",
        description="Ingest PDFs into synchronised FAISS + TF-IDF hybrid indexes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf-dir",
        default="pdfs",
        metavar="PATH",
        help="Directory containing source PDF files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="PATH",
        help="Root directory for index outputs.",
    )
    parser.add_argument(
        "--model",
        default="NeuML/pubmedbert-base-embeddings",
        metavar="NAME",
        dest="model_name",
        help="HuggingFace embedding model identifier.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        metavar="INT",
        help="Target chunk size in tokens.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        metavar="INT",
        help="Token overlap between consecutive same-section chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="INT",
        dest="embedding_batch_size",
        help="Embedding batch size (increase for GPU).",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=40,
        metavar="INT",
        help="Minimum token count for a content chunk to be indexed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return parser


def main() -> None:
    """CLI entry point — also callable as ``python -m rag_system.ingestion.DataIngestion``."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        settings = Settings(
            model_name=args.model_name,
            pdf_dir=Path(args.pdf_dir),
            output_dir=Path(args.output_dir),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_batch_size=args.embedding_batch_size,
            min_chunk_tokens=args.min_chunk_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        count = ingest(args.pdf_dir, args.output_dir, settings)
        print(f"✅  Indexed {count} chunks → {Path(args.output_dir) / 'faiss_index'}")
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

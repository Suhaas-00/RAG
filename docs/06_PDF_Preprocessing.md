# PDF Preprocessing

## Table of Contents
- [Library](#library)
- [Page Extraction](#page-extraction)
- [Cleaning Before Chunking](#cleaning-before-chunking)
- [OCR](#ocr)
- [Unicode Handling](#unicode-handling)
- [Failure Handling](#failure-handling)

## Library

PDF loading uses `pypdf.PdfReader`. `pdfplumber` is listed in `requirements.txt` but is not imported by current source code.

## Page Extraction

`load_pdf(path)` returns `PageText(page.extract_text() or "", page_number=number)` for every page, using 1-based page numbers. Empty extracted pages are retained as empty strings.

## Cleaning Before Chunking

`chunk_pages` calls `_pages_to_units`, which calls `clean_pdf_pages`. That cleaner removes repeated headers/footers, rejects low-quality lines, preserves section headings, and stops reading tail sections such as references and funding until a new recognized section heading appears.

## OCR

No OCR is implemented. Image-only scanned PDFs will not produce useful text unless `pypdf` can extract text.

## Unicode Handling

`normalize_text` and `clean_text` apply NFKC normalization, remove soft hyphen U+00AD, normalize CRLF/CR to LF, and rejoin soft-hyphen line breaks.

## Failure Handling

`load_pdf` raises `FileNotFoundError` for missing files and `ValueError` for files that `PdfReader` cannot open. `ingest` catches those per PDF and continues.

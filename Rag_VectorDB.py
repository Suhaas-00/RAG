"""Production-oriented PDF ingestion and local semantic search with LangChain + FAISS.

The index is rebuilt atomically when the corpus or configuration changes. This
keeps edited/deleted PDFs from leaving stale vectors in the database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


LOG = logging.getLogger("pdf_faiss")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Settings:
    pdf_dir: str
    index_dir: str
    model_name: str = DEFAULT_MODEL
    chunk_size: int = 450
    chunk_overlap: int = 75
    batch_size: int = 32
    device: str = "cpu"

    def validate(self) -> None:
        if self.chunk_size < 50:
            raise ValueError("chunk_size must be at least 50 tokens")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    return sorted(
        (path for path in pdf_dir.rglob("*.pdf") if path.is_file()),
        key=lambda path: path.as_posix().lower(),
    )


def corpus_manifest(pdf_dir: Path, paths: Sequence[Path], settings: Settings) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(pdf_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    identity = {
        "files": files,
        "model_name": settings.model_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**identity, "fingerprint": fingerprint}


def clean_pdf_text(text: str) -> str:
    """Remove extraction artifacts without destroying meaningful numbers/citations."""
    import re

    text = text.replace("\x00", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(path: Path, pdf_dir: Path) -> list[Document]:
    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:
        raise RuntimeError(f"Could not parse {path}: {exc}") from exc

    relative_source = path.relative_to(pdf_dir).as_posix()
    result: list[Document] = []
    for page_index, page in enumerate(pages):
        text = clean_pdf_text(page.page_content)
        if not text:
            continue
        metadata = dict(page.metadata)
        metadata.update(
            {
                "source": relative_source,
                "file_name": path.name,
                "page": int(metadata.get("page", page_index)),
                "file_sha256": sha256_file(path),
            }
        )
        result.append(Document(page_content=text, metadata=metadata))
    return result


def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
        add_start_index=True,
    )


def stable_chunk_id(document: Document) -> str:
    material = "\x1f".join(
        [
            str(document.metadata.get("source", "")),
            str(document.metadata.get("page", "")),
            str(document.metadata.get("start_index", "")),
            document.page_content,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def iter_chunks(
    paths: Sequence[Path], pdf_dir: Path, splitter: RecursiveCharacterTextSplitter, strict: bool
) -> Iterator[Document]:
    for position, path in enumerate(paths, start=1):
        LOG.info("Loading PDF %d/%d: %s", position, len(paths), path)
        try:
            chunks = splitter.split_documents(load_pdf_pages(path, pdf_dir))
        except Exception as exc:
            if strict:
                raise
            LOG.error("Skipping unreadable PDF: %s", exc)
            continue
        for chunk in chunks:
            chunk.metadata["chunk_id"] = stable_chunk_id(chunk)
            yield chunk


def make_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.model_name,
        model_kwargs={"device": settings.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.batch_size,
        },
    )


def batched(items: Sequence[Document], size: int) -> Iterator[Sequence[Document]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def write_jsonl(path: Path, documents: Iterable[Document]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {"text": document.page_content, "metadata": document.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def atomic_publish(temp_dir: Path, index_dir: Path) -> None:
    backup = index_dir.with_name(f"{index_dir.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    try:
        if index_dir.exists():
            index_dir.rename(backup)
        temp_dir.rename(index_dir)
    except Exception:
        if not index_dir.exists() and backup.exists():
            backup.rename(index_dir)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def read_manifest(index_dir: Path) -> dict[str, Any] | None:
    path = index_dir / "manifest.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ingest(settings: Settings, *, force: bool = False, strict: bool = False) -> int:
    settings.validate()
    pdf_dir = Path(settings.pdf_dir).expanduser().resolve()
    index_dir = Path(settings.index_dir).expanduser().resolve()
    paths = discover_pdfs(pdf_dir)
    if not paths:
        raise FileNotFoundError(f"No PDF files found under {pdf_dir}")

    candidate = corpus_manifest(pdf_dir, paths, settings)
    existing = read_manifest(index_dir)
    if not force and existing and existing.get("fingerprint") == candidate["fingerprint"]:
        LOG.info("Index is current; no ingestion needed: %s", index_dir)
        return int(existing.get("document_count", 0))

    splitter = build_splitter(settings)
    documents = list(iter_chunks(paths, pdf_dir, splitter, strict))
    if not documents:
        raise RuntimeError("No text chunks were produced from the PDF corpus")

    embeddings = make_embeddings(settings)
    ids = [str(doc.metadata["chunk_id"]) for doc in documents]
    vectorstore: FAISS | None = None
    for batch_number, batch in enumerate(batched(documents, settings.batch_size), start=1):
        LOG.info("Embedding batch %d", batch_number)
        offset = (batch_number - 1) * settings.batch_size
        batch_ids = ids[offset : offset + len(batch)]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(list(batch), embeddings, ids=batch_ids)
        else:
            vectorstore.add_documents(list(batch), ids=batch_ids)
    assert vectorstore is not None

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{index_dir.name}-", dir=index_dir.parent))
    try:
        vectorstore.save_local(str(temp_dir))
        write_jsonl(temp_dir / "chunks.jsonl", documents)
        manifest = {
            **candidate,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "document_count": len(documents),
            "settings": asdict(settings),
        }
        with (temp_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        atomic_publish(temp_dir, index_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    LOG.info("Indexed %d chunks from %d PDFs into %s", len(documents), len(paths), index_dir)
    return len(documents)


def load_index(settings: Settings) -> FAISS:
    index_dir = Path(settings.index_dir).expanduser().resolve()
    if not (index_dir / "index.faiss").is_file():
        raise FileNotFoundError(f"FAISS index not found: {index_dir}")
    # FAISS's local docstore uses pickle. Only load indexes created and controlled by you.
    return FAISS.load_local(
        str(index_dir),
        make_embeddings(settings),
        allow_dangerous_deserialization=True,
    )


def search(settings: Settings, query: str, k: int, fetch_k: int) -> list[Document]:
    store = load_index(settings)
    return store.max_marginal_relevance_search(query, k=k, fetch_k=max(fetch_k, k))


def print_stats(settings: Settings) -> None:
    manifest = read_manifest(Path(settings.index_dir).expanduser().resolve())
    if manifest is None:
        raise FileNotFoundError("No manifest found; run ingest first")
    summary = {
        "created_at": manifest.get("created_at"),
        "pdf_count": len(manifest.get("files", [])),
        "chunk_count": manifest.get("document_count", 0),
        "model_name": manifest.get("model_name"),
        "fingerprint": manifest.get("fingerprint"),
    }
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pdf-dir", default=os.getenv("PDF_DIR", "./pdf"))
    common.add_argument("--index-dir", default=os.getenv("INDEX_DIR", "./output/faiss_index"))
    common.add_argument("--model", default=os.getenv("EMBED_MODEL", DEFAULT_MODEL))
    common.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "450")))
    common.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "75")))
    common.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "32")))
    common.add_argument("--device", default=os.getenv("EMBED_DEVICE", "cpu"))

    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", parents=[common], help="Build/update the index")
    ingest_parser.add_argument("--force", action="store_true")
    ingest_parser.add_argument("--strict", action="store_true", help="Fail on the first unreadable PDF")
    search_parser = commands.add_parser("search", parents=[common], help="Search the index")
    search_parser.add_argument("query")
    search_parser.add_argument("-k", type=int, default=5)
    search_parser.add_argument("--fetch-k", type=int, default=20)
    commands.add_parser("stats", parents=[common], help="Show index metadata")
    return root


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        pdf_dir=args.pdf_dir,
        index_dir=args.index_dir,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        settings = settings_from_args(args)
        if args.command == "ingest":
            ingest(settings, force=args.force, strict=args.strict)
        elif args.command == "search":
            for rank, document in enumerate(search(settings, args.query, args.k, args.fetch_k), 1):
                source = document.metadata.get("source", "unknown")
                page = int(document.metadata.get("page", 0)) + 1
                print(f"\n[{rank}] {source}, page {page}\n{document.page_content}")
        elif args.command == "stats":
            print_stats(settings)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

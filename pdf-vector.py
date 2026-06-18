import os
import shutil
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb

# ====== CONFIG ======
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "multi_pdf_rag"


def read_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages_data = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_data.append({"text": text, "page": i + 1})
    return pages_data


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def reset_database(client_ref):
    """
    Close and dereference the ChromaDB client FIRST (releases the SQLite lock),
    then delete the folder on disk.
    """
    if client_ref is not None:
        try:
            # chromadb PersistentClient does not expose a .close() method,
            # but deleting the Python object releases the file handle on Windows.
            del client_ref
        except Exception:
            pass

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 Old ChromaDB deleted")


def process_multiple_pdfs(folder_path, collection, model):
    all_chunks = []
    all_metadata = []
    all_ids = []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    print(f"📂 Found {len(pdf_files)} PDFs")

    if not pdf_files:
        print("❌ No PDF files found in folder:", folder_path)
        return

    global_id = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\n📄 Processing: {pdf_file}")

        pages = read_pdf(pdf_path)

        for page_data in pages:
            chunks = chunk_text(page_data["text"])

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append(
                    {
                        "source": pdf_file,
                        "page": page_data["page"],
                        "chunk_id": i,
                    }
                )
                all_ids.append(f"id_{global_id}")
                global_id += 1

    print(f"\n✂️  Total chunks created: {len(all_chunks)}")

    if not all_chunks:
        print("❌ No text extracted from PDFs!")
        return

    # Generate embeddings
    print("🔄 Generating embeddings...")
    embeddings = model.encode(all_chunks).tolist()

    # Store in ChromaDB
    print("💾 Storing in ChromaDB...")
    collection.add(
        documents=all_chunks,
        metadatas=all_metadata,
        embeddings=embeddings,
        ids=all_ids,
    )

    print("✅ Stored successfully!")
    print("📊 Total items in DB:", collection.count())
    print("📁 DB location:", os.path.abspath(CHROMA_PATH))
    print("Dimensions of your Embeddings ",len(embeddings[0]))  


if __name__ == "__main__":
    folder = "pdfs"

    # ── Step 1: destroy any existing client object BEFORE deleting the folder ──
    # (At module start there is no client yet, so we pass None.)
    reset_database(client_ref=None)

    # ── Step 2: create a fresh client AFTER the folder has been removed ────────
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # ── Step 3: load the embedding model ──────────────────────────────────────
    print("📦 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Step 4: ingest ────────────────────────────────────────────────────────
    process_multiple_pdfs(folder, collection, model)
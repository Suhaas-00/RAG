import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB (local persistent DB)
client = chromadb.Client(Settings(
    persist_directory="./chroma_db"
))

# Create / get collection
collection = client.get_or_create_collection(name="multi_pdf_rag")


def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)

        texts = []
        total_pages = len(reader.pages)

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                texts.append({
                    "text": text,
                    "page": i + 1
                })

    return texts, total_pages


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def process_multiple_pdfs(folder_path):
    all_chunks = []
    all_metadata = []
    all_ids = []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    print(f"📂 Found {len(pdf_files)} PDFs")

    global_id = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\n📄 Processing: {pdf_file}")

        pages, total_pages = read_pdf(pdf_path)

        full_text = " ".join([p["text"] for p in pages])
        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            metadata = {
                "source": pdf_file,
                "chunk_id": i
            }
            all_metadata.append(metadata)

            all_ids.append(f"id_{global_id}")
            global_id += 1

    print(f"\n✂️ Total chunks: {len(all_chunks)}")

    # Generate embeddings
    print("🔄 Generating embeddings...")
    embeddings = model.encode(all_chunks).tolist()

    # Store in ChromaDB
    print("💾 Storing in ChromaDB...")
    collection.add(
        documents=all_chunks,
        metadatas=all_metadata,
        embeddings=embeddings,
        ids=all_ids
    )

    # Persist DB
    client.persist()

    print("✅ Multi-PDF stored in ChromaDB successfully!")


if __name__ == "__main__":
    folder = "pdfs"
    process_multiple_pdfs(folder)
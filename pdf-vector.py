import os
import faiss
import PyPDF2
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


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

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    print(f"📂 Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\n📄 Processing: {pdf_file}")

        pages, total_pages = read_pdf(pdf_path)

        full_text = " ".join([p["text"] for p in pages])

        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            all_metadata.append({
                "source": pdf_file,          # ✅ IMPORTANT
                "chunk_id": i,
                "text": chunk
            })

    print(f"\n✂️ Total chunks: {len(all_chunks)}")

    # Generate embeddings
    print("🔄 Generating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save
    faiss.write_index(index, "multi_vectors.index")

    with open("multi_chunks.pkl", "wb") as f:
        pickle.dump({
        "chunks": all_chunks,
        "metadata": all_metadata,
        "embeddings": embeddings   # 🔥 ADD THIS
        }, f)

    print("✅ Multi-PDF index created successfully!")


if __name__ == "__main__":
    folder = "pdfs"   # 📂 Put all PDFs here
    process_multiple_pdfs(folder)
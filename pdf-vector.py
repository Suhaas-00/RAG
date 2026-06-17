import faiss
import PyPDF2
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def pdf_to_vectors(pdf_path):
    print(f"📄 Reading PDF: {pdf_path}")

    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)

        page_texts = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                page_texts.append({
                    'text': text,
                    'page_number': page_num + 1
                })

    # Combine text
    text = ''.join([p['text'] for p in page_texts])

    print(f"📊 Total pages: {total_pages}")
    print(f"📊 Total text length: {len(text):,}")

    # Chunking
    chunks = []
    metadata = []

    for i in range(0, len(text), 400):
        chunk = text[i:i + 500]
        chunks.append(chunk)

        estimated_page = min((i // (len(text) // total_pages)) + 1, total_pages)

        metadata.append({
            "text": chunk,
            "estimated_page": estimated_page
        })

    print(f"✂️ Created {len(chunks)} chunks")

    # Embeddings
    print("🔄 Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Normalize
    faiss.normalize_L2(embeddings)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, "vectors.index")

    # ✅ FIX: Save COMPLETE STRUCTURE
    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "metadata": metadata,
            "total_pages": total_pages
        }, f)

    print("✅ Index created successfully!")
    print("📁 Files saved: vectors.index, chunks.pkl")


if __name__ == "__main__":
    pdf_file = "MongoDB.pdf"  # ✅ Make sure file exists in same folder
    pdf_to_vectors(pdf_file)
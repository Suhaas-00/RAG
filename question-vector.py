import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
# Load embedding model (MUST match indexing)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load variables from the .env file into the system environment
load_dotenv()

# Initialize Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_question(question):
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Vector database not found!")
        print("🔧 Run: python create_index.py")
        return None

    try:
        # Load data
        index = faiss.read_index("vectors.index")

        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']

        # Encode query
        query_vector = model.encode([question])
        query_vector = np.array(query_vector).astype('float32')

        # Normalize
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = index.search(query_vector, 3)

        print(f"\n🔍 Found {len(indices[0])} relevant chunks:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            page = metadata[idx]['estimated_page']
            print(f"   {i+1}. Score: {score:.3f} (Page ~{page})")

        # Build context
        context_parts = []
        for idx in indices[0]:
            chunk_text = chunks[idx]
            page = metadata[idx]['estimated_page']
            context_parts.append(f"[Page {page}]: {chunk_text}")

        context = "\n\n".join(context_parts)

        # Groq LLM
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"You are answering questions about a {total_pages}-page document. Mention page numbers when relevant."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Vector DB not found!")
        print("Run: python create_index.py")
        return

    # Load info
    try:
        index = faiss.read_index("vectors.index")

        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        total_pages = data['total_pages']

        print(f"✅ Loaded: {len(chunks)} chunks from {total_pages} pages")

    except Exception as e:
        print(f"❌ Load error: {str(e)}")
        return

    print("\n" + "="*60)
    print("🤖 RAG System Ready")
    print("Type 'exit' to quit | 'info' for stats")
    print("="*60)

    while True:
        question = input("\n❓ Question: ").strip()

        if question.lower() in ['exit', 'quit', 'q']:
            print("👋 Exiting...")
            break

        if question.lower() == 'info':
            print("\n📊 Info:")
            print(f"Pages: {total_pages}")
            print(f"Chunks: {len(chunks)}")
            print(f"Dimension: {index.d}")
            print(f"Avg chunks/page: {len(chunks)/total_pages:.2f}")
            continue

        if not question:
            print("⚠️ Enter a valid question")
            continue

        print("🔍 Processing...")
        answer = ask_question(question)

        if answer:
            print(f"\n🤖 Answer:\n{answer}")
        else:
            print("❌ Failed to generate answer")


if __name__ == "__main__":
    main()
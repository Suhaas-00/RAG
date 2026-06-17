import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import chromadb

# ====== CONFIG ======
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "multi_pdf_rag"

# Load env
load_dotenv()

# Load models
print("📦 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("🔑 Loading Groq client...")
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ✅ CORRECT: Load persistent DB
print("📂 Connecting to ChromaDB...")
client_db = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client_db.get_or_create_collection(name=COLLECTION_NAME)


def ask_question(question):
    try:
        print("\n🚀 Processing question...")

        db_count = collection.count()
        print("📊 DB Count:", db_count)

        if db_count == 0:
            return "❌ Database is empty. Run pdf-vector.py first."

        # Clean query
        query = question.strip()

        # Encode query
        query_embedding = model.encode([query]).tolist()

        # Search
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return "❌ No relevant information found."

        print("\n🔍 Top Matches:")
        for meta, score in zip(metadatas, distances):
            print(f"{meta['source']} (Page {meta['page']}) → {score:.4f}")

        # Build context
        context = "\n\n".join([
            f"[Source: {meta['source']} | Page: {meta['page']}]\n{doc}"
            for doc, meta in zip(documents, metadatas)
        ])

        # LLM call
        response = client_llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Answer ONLY using the provided context. Mention source and page."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"


if __name__ == "__main__":
    print("=" * 50)
    print("🤖 RAG System (ChromaDB) Ready")
    print("=" * 50)

    while True:
        q = input("\n❓ Enter your question: ").strip()

        if q.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        if not q:
            continue

        answer = ask_question(q)
        print("\n🤖 Answer:\n", answer)
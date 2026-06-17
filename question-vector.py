import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# Load env
load_dotenv(override=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
client_llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load ChromaDB
client_db = chromadb.Client(Settings(
    persist_directory="./chroma_db"
))

collection = client_db.get_or_create_collection(name="multi_pdf_rag")


def ask_question(question):
    try:
        print("🚀 Processing question...")

        # 🔥 Improve query
        question = question.lower()

        # Encode query
        query_embedding = model.encode([question]).tolist()

        # 🔍 Search in ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print("\n🔍 Top Matches:")
        for meta, score in zip(metadatas, distances):
            print(f"{meta['source']} → {score:.3f}")

        # Build context
        context_parts = []
        for doc, meta in zip(documents, metadatas):
            context_parts.append(
                f"[Source: {meta['source']} | Chunk: {meta['chunk_id']}]\n{doc}"
            )

        context = "\n\n".join(context_parts)

        # LLM
        response = client_llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Answer ONLY from the context. Mention source document."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("🤖 Multi-PDF RAG System (ChromaDB) Ready")
    print("=" * 50)

    while True:
        q = input("\n❓ Enter your question: ").strip()

        if q.lower() in ["exit", "quit"]:
            break

        if not q:
            continue

        answer = ask_question(q)

        if answer:
            print("\n🤖 Answer:\n", answer)
        else:
            print("❌ Failed to generate answer")
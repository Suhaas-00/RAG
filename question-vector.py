import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load env
load_dotenv(override=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_question(question):
    if not os.path.exists("multi_vectors.index") or not os.path.exists("multi_chunks.pkl"):
        print("❌ Vector database not found!")
        return None

    try:
        print("🚀 Processing question...")

        # Load
        index = faiss.read_index("multi_vectors.index")

        with open("multi_chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']

        # 🔥 Improve query
        question = question.lower()

        # Encode query
        query_vector = model.encode([question])
        query_vector = np.array(query_vector).astype('float32')
        faiss.normalize_L2(query_vector)

        # 🔥 GLOBAL SEARCH (NO PDF FILTERING)
        scores, indices = index.search(query_vector, 5)

        print("\n🔍 Top Matches:")
        for score, idx in zip(scores[0], indices[0]):
            print(f"{metadata[idx]['source']} → {score:.3f}")

        # Build context
        context_parts = []
        for idx in indices[0]:
            context_parts.append(
                f"[Source: {metadata[idx]['source']}]\n{chunks[idx]}"
            )

        context = "\n\n".join(context_parts)

        # LLM
        response = client.chat.completions.create(
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
    print("🤖 Multi-PDF RAG System Ready")
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
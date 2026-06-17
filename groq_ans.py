from groq import Groq
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load variables from the .env file into the system environment
load_dotenv()

# Initialize Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("vectors.index")

with open("chunks.pkl", "rb") as f:
    metadata = pickle.load(f)


def retrieve(query, top_k=5):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    context = "\n\n".join([metadata[i]['text'] for i in indices[0]])
    return context


def ask_groq(query):
    context = retrieve(query)

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        query = input("\n❓ Ask: ")
        if query.lower() == "exit":
            break

        answer = ask_groq(query)
        print("\n🤖 Answer:\n", answer)
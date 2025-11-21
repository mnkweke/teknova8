from flask import Flask, render_template, request, jsonify
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Load environment variable (Render → Environment → Add "GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

# Load embeddings model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(text):
    return embedding_model.encode(text).astype(np.float32)

# Load your FAISS index
vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")

    # Retrieve similar chunks
    results = vectorstore.similarity_search(user_message, k=3)
    context = "\n\n".join([item.page_content for item in results])

    prompt = f"Context:\n{context}\n\nUser question:\n{user_message}\n\nAnswer:"

    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192"
    )

    reply = completion.choices[0].message["content"]

    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

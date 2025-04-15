import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

import google.generativeai as genai
# ========== SETUP ==========

# Load Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # <-- Replace with your key
model_gemini = genai.GenerativeModel("gemini-1.5-pro-latest")  # Or a more suitable model

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# Load FAISS index and chunks
index = faiss.read_index("chunk_index.faiss")
with open("chunk_texts.pkl", "rb") as f:
    chunks = pickle.load(f)

# ========== FUNCTIONS ==========

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().numpy()

def query_faiss(query, top_k=3):
    query_vector = get_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

def get_answer_from_gemini(query, context_chunks,max_output_tokens=200):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are an assistant with access to the following context:\n\n{context}\n\nUser question: {query}\n\nPlease answer the question based only on the context above."""
    global model_gemini
    response = model_gemini.generate_content(prompt)
    return response.text

def search_and_respond(user_query):
    try:
        top_chunks = query_faiss(user_query,max_output_tokens=200)
        response = get_answer_from_gemini(user_query, top_chunks,max_output_tokens=max_output_tokens)
        return response
    except Exception as e:
        return f"Failed to process query: {str(e)}"

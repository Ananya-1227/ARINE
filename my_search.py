import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import time
import google.api_core.exceptions  # Make sure this is imported
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from deepseek_api import deepseek_answer

import google.generativeai as genai
# ========== SETUP ==========

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/generate"

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index and chunks
index = faiss.read_index("chunk_index.faiss")
with open("chunk_texts.pkl", "rb") as f:
    chunks = pickle.load(f)

# ========== FUNCTIONS ==========

def get_embedding(text):
    """
    Get embedding for a given text using a transformer-based model (MiniLM in this case).
    Returns: numpy array of the embedding.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().numpy()

def query_faiss(query, top_k=2):
    """
    Query the FAISS index to find the top-k relevant chunks for the user's query.
    Returns: List of the most relevant chunks.
    """
    query_vector = get_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

def get_answer_from_deepseek(query, context_chunks, max_output_tokens=200, retries=3):
    """
    Get an answer from the DeepSeek API based on the provided query and context chunks.
    Handles retries on failure and returns the response text.
    """
    context = "\n\n".join(context_chunks)
    prompt = f"""You are an assistant with access to the following context:\n\n{context}\n\nUser question: {query}\n\nPlease answer the question based only on the context above."""

    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json',
    }

    data = {
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
    }

    # Retry logic for handling potential API issues
    for attempt in range(retries):
        try:
            response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Assuming DeepSeek API returns a JSON response with a 'text' field
            answer = response.json().get("text", "No response text found.")
            return answer

        except requests.exceptions.Timeout:
            print(f"[Attempt {attempt+1}] Timeout: Retrying in 2s...")
            time.sleep(2)
        
        except requests.exceptions.RequestException as e:
            print(f"[Attempt {attempt+1}] RequestException: {e}")
            return f"DeepSeek error: {str(e)}"

    return "DeepSeek API timed out after multiple attempts. Try a shorter query or fewer context chunks."

def search_and_respond(user_query, max_output_tokens=200, top_k=2):
    """
    Main function to handle the user query, query the FAISS index, and get a response from DeepSeek.
    Allows dynamic adjustments for max output tokens and the number of top-k context chunks.
    """
    try:
        # Get the top-k most relevant context chunks for the query
        top_chunks = query_faiss(user_query, top_k=top_k)
        
        # Get an answer from DeepSeek based on the top chunks
        response = get_answer_from_deepseek(user_query, top_chunks, max_output_tokens=max_output_tokens)
        
        return response
    except Exception as e:
        return f"Failed to process query: {str(e)}"

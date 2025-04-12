import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to convert text to embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().numpy()

# Load the FAISS index and text chunks
index = faiss.read_index("chunk_index.faiss")
with open("chunk_texts.pkl", "rb") as f:
    chunks = pickle.load(f)

# Function to query the FAISS index
def query_faiss(query, top_k=3):
    query_vector = get_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, 1)
    results = chunks[indices[0][0]]
    return results


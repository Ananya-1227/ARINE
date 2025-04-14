import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import re
import google.generativeai as genai
from dotenv import load_dotenv
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# --- 2.  Modified Gemini Function ---
def gemini_answer(prompt: str, context: str, max_tokens=500) -> str:
    """
    Gets an answer from the Gemini model, using the provided context.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Use correct model
        prompt_with_context = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        response = model.generate_content(prompt_with_context,
                                        generation_config=genai.types.GenerationConfig(
                                            # max_output_tokens=max_tokens,
                                            temperature=0.3,
                                            # top_p=0.7,
                                            # stop_sequences=["\n\n"]
                                        ))
        clean_text = re.sub(r'<[^>]+>', '', response.text)
        clean_text = re.sub(r'[\*\_]', '', clean_text)  # Remove * and _
        return clean_text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def gemini_summarize(text: str, max_tokens: int = 300) -> str:
    """
    Summarizes the given text using Gemini.
    """

    model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Or a more suitable model
    prompt = f"Summarize the following text:\n{text}"
    try:
        response = model.generate_content(prompt,
                                        generation_config=genai.types.GenerationConfig(
                                            max_output_tokens=max_tokens,
                                            temperature=0.4,
                                            top_p=0.8
                                        ))
        clean_text = re.sub(r'<[^>]+>', '', response.text)
        clean_text = re.sub(r'[\*\_]', '', clean_text)
        return clean_text.strip()
    except Exception as e:
        return f"Error: {str(e)}"
        
def search_chunks(query: str, index: faiss.Index, chunks: List[str], top_k: int = 3) -> str:
    query_embedding = embedding_model.encode([query])  # Embed the query
    D, I = index.search(query_embedding, top_k)  # Search the index

    context_chunks = [chunks[i] for i in I[0]]  # Get the text chunks
    context = "\n\n".join(context_chunks)  # Combine into a context string

    return context
    


# # Function to convert text to embedding
# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state
#     input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
#     sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     return (sum_embeddings / sum_mask).squeeze().numpy()

# # Load the FAISS index and text chunks
# index = faiss.read_index("chunk_index.faiss")
# with open("chunk_texts.pkl", "rb") as f:
#     chunks = pickle.load(f)

# # Function to query the FAISS index
# def query_faiss(query, top_k=3):
#     query_vector = get_embedding(query).astype("float32").reshape(1, -1)
#     distances, indices = index.search(query_vector, 1)
#     results = chunks[indices[0][0]]
#     return results


import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Ensure this key is set in your environment

def gemini_answer(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-pro")  # Use the correct model name
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {str(e)}"

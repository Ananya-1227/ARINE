import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Ensure this key is set in your environment

def gemini_answer(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Use the correct model name
        response = model.generate_content(prompt)
        return f'<span style="color:black;">{response.text.strip()}</span>'
    except Exception as e:
        return f'<span style="color:black;">Gemini Error: {str(e)}</span>'

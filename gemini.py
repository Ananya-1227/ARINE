import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Ensure this key is set in your environment

def gemini_answer(prompt: str,max_tokens=200) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Use the correct model name
        response = model.generate_content(prompt,generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,
            top_p=0.7,
            stop_sequences=["\n\n"]
            )
        )
        clean_text=re.sub(r'<[^>]+>','',response.text)
        clean_text = re.sub(r'[\*\_]', '', clean_text)  # Remove * and stuff
        return clean_text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

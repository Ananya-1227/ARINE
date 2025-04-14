import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def gemini_answer(prompt):
    # Get the API key from the environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ API key is missing. Please check your .env file."

    # ✅ CORRECT MODEL: Use 'gemini-pro' (text model) or 'gemini-pro-vision' (multimodal)
    model_id = "models/gemini-pro"  

    # ✅ CORRECT API ENDPOINT (v1beta)
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_id}:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    # Prepare the request data
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    # Send the POST request to Gemini API
    response = requests.post(url, headers=headers, json=data)

    # Check for success
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"❌ Gemini Error: {response.status_code} {response.text}"

# Test the function
if __name__ == "__main__":
    test_prompt = "Explain quantum computing in simple terms."
    print(gemini_answer(test_prompt))

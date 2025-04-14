import os
import requests
from dotenv import load_dotenv

load_dotenv()

def gemini_answer(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ API key missing. Check .env file."

    # ✅ Use the correct model name (Gemini 1.5 Pro or Gemini 1.0 Pro)
    model_id = "gemini-pro"  # or "gemini-1.5-pro-latest" for newer models

    # ✅ Updated API endpoint (v1beta or v1)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"❌ Gemini Error: {response.status_code} {response.text}"

# Test
if __name__ == "__main__":
    print(gemini_answer("How to become a HyperAchiever?"))

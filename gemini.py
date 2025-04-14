import os
import requests
from dotenv import load_dotenv

load_dotenv()

def gemini_answer(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ API key not found. Please set GEMINI_API_KEY in .env file."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"❌ Error parsing response: {str(e)}"
    else:
        return f"❌ Gemini Error: {response.status_code} {response.text}"

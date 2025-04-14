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

    # The correct API endpoint for Gemini
    # You should replace 'models/gemini-pro' with the correct model ID after listing available models
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{your_model_id}:generateContent?key={api_key}"

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


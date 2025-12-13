import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List models available for your key
models = genai.list_models()
for m in models:
    print(m)

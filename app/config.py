# app/config.py
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key_str = os.getenv('GEMINI_API_KEY', "")
if not api_key_str:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")

google_api_key = SecretStr(api_key_str)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["CONNECTION_STRING"]


def get_ollama_api_key() -> str:
    return os.environ["OLLAMA_API_KEY"]

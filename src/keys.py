import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
OPEN_AI_TOKEN = os.getenv("OPEN_AI_API_KEY")

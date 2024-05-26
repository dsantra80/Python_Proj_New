import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_NAME = os.getenv("MODEL_NAME")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
    TEMPERATURE = float(os.getenv("TEMPERATURE"))

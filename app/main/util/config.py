import os
from pathlib import Path
from dotenv import load_dotenv

# Charge .env à la racine du projet
BASE_DIR = Path(__file__).resolve().parents[3]  # arrive à la racine (où est .env)
load_dotenv(dotenv_path=BASE_DIR / ".env")

def get_env(key: str, default=None):
    return os.getenv(key, default)

def get_int(key: str, default=0):
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default

def get_float(key: str, default=0.0):
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default

# Expositions spécifiques
def embedding_model_name():
    return get_env("EMBEDDING_MODEL")

def llm_model_name():
    return get_env("LLM_MODEL_NAME")

def llm_model_token():
    return get_env("LLM_MODEL_TOKEN")

def vector_db_path():
    return get_env("DB_VECTOR_FOLDER", "db")

def top_k_default():
    return get_int("TOP_K_DEFAULT", 4)

def score_threshold():
    return get_float("SCORE_THRESHOLD", 0.0)

def service_api_key():
    return get_env("SERVICE_API_KEY")

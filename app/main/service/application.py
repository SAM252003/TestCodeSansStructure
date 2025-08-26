import os
from pathlib import Path

from flask import Flask, request
from flask_restx import Api
from dotenv import load_dotenv
from loguru import logger

# Charger .env (à partir de la racine du projet)
BASE_DIR = Path(__file__).resolve().parents[2]  # app/main/service/ -> remonte à app/main/
load_dotenv(dotenv_path=BASE_DIR.parent / ".env")  # .env à la racine

# Création de l'app
app = Flask(__name__)

# Configuration centrale
app.config["DB_VECTOR_FOLDER"] = os.getenv("DB_VECTOR_FOLDER", "db")
app.config["MODELS_FOLDER"] = os.getenv("MODELS_FOLDER", "models")
app.config["LLM_MODEL_NAME"] = os.getenv("LLM_MODEL_NAME", "")
app.config["LLM_MODEL_TOKEN"] = os.getenv("LLM_MODEL_TOKEN", "")
app.config["SCORE_THRESHOLD"] = float(os.getenv("SCORE_THRESHOLD", 0.0))
app.config["TOP_K_DEFAULT"] = int(os.getenv("TOP_K_DEFAULT", 4))

# Auth Bearer (facultatif) via SERVICE_API_KEY
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")

api = Api(app, version="1.0", title="RAG API", description="Knowledge assistant API")

# Import des namespaces après création de l'API pour éviter les boucles
from app.main.controller.ingest_controller import api as ingest_ns
from app.main.controller.generator_controller import api as generate_ns

# Montage des routes
api.add_namespace(ingest_ns, path="/ingest")
api.add_namespace(generate_ns, path="/generate")

# Middleware simple d'authentification Bearer
@app.before_request
def check_auth():
    pass

if __name__ == "__main__":
    logger.info("Starting RAG Flask service...")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    app.run(host=host, port=port, debug=True)

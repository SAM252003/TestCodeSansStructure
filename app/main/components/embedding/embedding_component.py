import os
import numpy as np
from typing import List
from loguru import logger
from dotenv import load_dotenv

# charge .env si ce n'est pas déjà fait ailleurs
from pathlib import Path
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

# import dynamique pour éviter import inutile
try:
    import ollama
except ImportError:  # Ollama peut ne pas être installé
    ollama = None

# OpenAI client (version >=1.0)
_openai_client = None
if os.getenv("EMBEDDING_MODEL", "").startswith("text-embedding"):
    try:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except ImportError:
        logger.warning("openai-python non installé, impossible d'utiliser OpenAI embeddings")


class EmbeddingComponent:
    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL")
        if not self.model_name:
            raise ValueError("EMBEDDING_MODEL non défini dans .env")
        self.logger = logger

        # Déterminer le backend
        if self.model_name.startswith("text-embedding") and _openai_client:
            self.backend = "openai"
            self.client = _openai_client
            self.logger.info(f"EmbeddingComponent using OpenAI model {self.model_name}")
        elif ollama is not None:
            self.backend = "ollama"
            self.logger.info(f"EmbeddingComponent using Ollama model {self.model_name}")
        else:
            raise RuntimeError("Aucun backend d'embedding disponible (ni OpenAI ni Ollama).")

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Retourne une liste d'embeddings numpy pour chaque texte.
        """
        if not isinstance(texts, list):
            raise ValueError("embed attend une liste de chaînes")

        if self.backend == "openai":
            resp = self.client.embeddings.create(model=self.model_name, input=texts)
            embeddings = []
            for d in resp.data:
                emb = np.array(d.embedding, dtype=np.float32)
                embeddings.append(emb)
            return embeddings

        elif self.backend == "ollama":
            resp = ollama.embed(model=self.model_name, input=texts)
            raw = resp.get("embeddings") or []
            return [np.array(v, dtype=np.float32) for v in raw]

        else:
            raise RuntimeError("Backend d'embedding inconnu")

    def embed_one(self, text: str) -> np.ndarray:
        """
        Shortcut pour un seul texte.
        """
        return self.embed([text])[0]

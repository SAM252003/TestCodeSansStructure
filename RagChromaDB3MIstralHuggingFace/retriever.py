# retriever.py
import os, numpy as np, chromadb, ollama
from chromadb.config import Settings
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


COLLECTION_NAME = "LivreBlancAkuiteo"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LANGUAGE_MODEL  = os.getenv("LANGUAGE_MODEL",  "llama3.2:1b-instruct-fp16")
TOP_K_DEFAULT   = int(os.getenv("TOP_K_DEFAULT", 4))
CHROMA_PATH     = os.getenv("CHROMA_PATH", "db")


client     = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve_chunks(query: str, top_k: int = 4):
    q_emb = np.array(
        ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0],
        dtype=np.float32
    )
    res = collection.query(query_embeddings=[q_emb],
                           n_results=top_k,
                           include=["documents", "metadatas", "distances"])
    return [
        {
            "section": m["section"],
            "text": d,
            "score": 1 - dist        # convert distance → similarity
        }
        for d, m, dist in zip(res["documents"][0],
                              res["metadatas"][0],
                              res["distances"][0])
    ]

import os, numpy as np, chromadb, openai
from chromadb.config import Settings
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

COLLECTION_NAME  = "LivreBlancAkuiteo"
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
LANGUAGE_MODEL   = os.getenv("LANGUAGE_MODEL", "gpt-3.5-turbo-0125")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K_DEFAULT    = int(os.getenv("TOP_K_DEFAULT", 4))
CHROMA_PATH      = os.getenv("CHROMA_PATH", "db")

client     = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def get_embedding(text: str) -> np.ndarray:
    response = openai.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)
# retriever.py

def retrieve_chunks(query: str, top_k: int = TOP_K_DEFAULT):
    q_emb = get_embedding(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist
        item = {"score": sim}
        if meta.get("type") == "image":
            item.update({
                "type":    "image",
                "caption": doc,
                "page":    meta["page"],
                "path":    meta["path"]
            })
        else:
            item.update({
                "type":    "text",
                "section": meta["section"],
                "text":    doc
            })
        results.append(item)
    return results

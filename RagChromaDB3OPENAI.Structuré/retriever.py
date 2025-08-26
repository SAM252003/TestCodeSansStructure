import os, numpy as np, chromadb, openai
from chromadb.config import Settings
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

COLLECTION_NAME  = "LivreBlancAkuiteo"
openai.api_key = os.getenv("OPENAI_API_KEY")

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
def retrieve_chunks(query: str, top_k: int = TOP_K_DEFAULT):
    q_emb = get_embedding(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "section": m["section"],
            "text": d,
            "score": 1 - dist  # distance -> similarité
        }
        for d, m, dist in zip(res["documents"][0],
                              res["metadatas"][0],
                              res["distances"][0])
    ]


if __name__ == "__main__":
    for q in [
        "Créer un bloc planning",
        "9.3.4 Créer un bloc de planning",
        "Comment créer un bloc planning"
    ]:
        print("Query:", q)
        hits = retrieve_chunks(q, top_k=3)
        print("→", hits, "\n")
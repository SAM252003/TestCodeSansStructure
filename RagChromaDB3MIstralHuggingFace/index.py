import os, numpy as np
from pathlib import Path
import chromadb, ollama
from chromadb.config import Settings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from loader_markitdown import load_livre_dor
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


COLLECTION_NAME = "LivreBlancAkuiteo"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "db")

def embed(texts):
    res = ollama.embed(model=EMBEDDING_MODEL, input=texts)["embeddings"]
    return [np.array(v, dtype=np.float32) for v in res]

def main(pdf_file: str):
    raw_docs = load_livre_dor(pdf_file)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")],
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
    col    = client.get_or_create_collection(name=COLLECTION_NAME)

    BATCH = 256
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        col.add(
            documents=[d.page_content for d in batch],
            metadatas=[d.metadata for d in batch],
            embeddings=embed([d.page_content for d in batch]),
            ids=[d.metadata["id"] for d in batch]
        )
        print(col.count())



import os, numpy as np
from pathlib import Path
import chromadb, ollama
from chromadb.config import Settings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from loader_markitdown import load_livre_dor
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
import uuid


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
COLLECTION_NAME = "LivreBlancAkuiteo"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "../db")


def embed(texts):
    res = ollama.embed(model=EMBEDDING_MODEL, input=texts)["embeddings"]
    return [np.array(v, dtype=np.float32) for v in res]


def main(pdf_file: str):
    raw_docs = load_livre_dor(pdf_file)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    header_chunks: list[Document] = []

    for doc in raw_docs:
        # Tenter split en texte brut
        sections = header_splitter.split_text(doc.page_content)

        # Si ça revient sous forme de Document, on gère aussi
        if sections and isinstance(sections[0], Document):
            # on récupère directement ces Documents, en fusionnant les métadonnées
            for sec in sections:
                sec.metadata.update(doc.metadata)
                header_chunks.append(sec)
        else:
            # on a bien une liste de str
            for txt in sections:
                header_chunks.append(
                    Document(page_content=txt, metadata=doc.metadata)
                )

    #  Split de chaque chunk en morceaux de taille limitée
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = chunk_splitter.split_documents(header_chunks)

    #  Ingestion dans ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(allow_reset=True)
    )
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    BATCH = 256
    for batch_idx in range(0, len(chunks), BATCH):
        batch = chunks[batch_idx: batch_idx + BATCH]
        # Génère un ID unique pour chaque chunk
        ids = [
            f"{d.metadata['id']}_{batch_idx + idx}"
            for idx, d in enumerate(batch)
        ]

        col.add(
            documents=[d.page_content for d in batch],
            metadatas=[d.metadata for d in batch],
            embeddings=embed([d.page_content for d in batch]),
            ids=ids,
        )
    # 5️⃣ Afficher le compte
    print("Nombre de documents dans la collection :", col.count())

# Exécution réelle
if __name__ == "__main__":
    main("/Users/samsebag/PycharmProjects/LangChain /RagChromaDB3/data/LivreDor.pdf")
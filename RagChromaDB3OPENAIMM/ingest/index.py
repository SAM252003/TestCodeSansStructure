# ingest/index_multimodal.py

from extract_images import extract_images, ocr_image
from loader_markitdown import load_livre_dor
from chromadb.config import Settings
import os, numpy as np, chromadb, openai
from pathlib import Path
from langchain.text_splitter import MarkdownHeaderTextSplitter


def main(pdf_file: str):
    raw_docs = load_livre_dor(pdf_file)

    # 2) Extraire & OCR des images
    images = extract_images(pdf_file)
    ocr_by_page: dict[int, list[str]] = {}
    for img in images:
        text = ocr_image(img["path"])
        ocr_by_page.setdefault(img["page"], []).append(text)

    # 3) Split en chunks de texte (_sans_ images pour le moment)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(raw_docs)

    # 4) Pour chaque chunk, on récupère son page, on y ajoute les OCR
    enriched_chunks = []
    for chunk in chunks:
        page = chunk.metadata.get("page")
        ocr_texts = ocr_by_page.get(page, [])
        if ocr_texts:
            # on ajoute les descriptions d'image _à la fin_ du chunk
            combined = chunk.page_content + "\n\n" + "\n\n".join(ocr_texts)
        else:
            combined = chunk.page_content
        # on crée un nouveau Document-like dict pour l’index
        enriched_chunks.append({
            "text": combined,
            "id":    chunk.metadata["id"],
            "metadata": {**chunk.metadata, "has_images": bool(ocr_texts)}
        })

    # 5) Indexation dans Chroma
    client     = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    BATCH = 256
    for i in range(0, len(enriched_chunks), BATCH):
        batch = enriched_chunks[i : i + BATCH]
        texts  = [e["text"] for e in batch]
        ids    = [e["id"]   for e in batch]
        metas  = [e["metadata"] for e in batch]
        embs   = get_embeddings(texts)
        collection.add(
            documents=texts,
            metadatas=metas,
            embeddings=embs,
            ids=ids
        )
        print(f"✓ {i + len(batch)}/{len(enriched_chunks)}")

    print("Indexation multimodale terminée.")


# Exécution réelle
if __name__ == "__main__":
    main("/Users/samsebag/PycharmProjects/LangChain /RagChromaDB3OPENAI/data/LivreDor.pdf")
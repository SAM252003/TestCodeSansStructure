import os
import json
import uuid
from pathlib import Path
from typing import Dict

from flask import current_app as app
from loguru import logger

from app.main.components.ingest.ingest_component import IngestComponent
from app.main.components.embedding.embedding_component import EmbeddingComponent
from app.main.components.vector_store.vector_store_component import VectorStoreComponent


class IngestService:
    def __init__(self):
        self.logger = logger
        self.ingester = IngestComponent()
        self.embedding_comp = EmbeddingComponent()
        self.vector_store = VectorStoreComponent(self.embedding_comp)

    def ingest_file(self, file_metadata: Dict, file) -> Dict:
        """
        Expose un point d'entrée pour /ingest : reçoit metadata + fichier,
        l'enregistre temporairement, extrait Documents, embedd, upsert.
        Attendu file_metadata: {"file_path": "...", "file_id": "..."} (file_path est utilisé comme nom/source)
        """
        try:
            # Préparation du fichier sur disque (temp)
            upload_dir = Path(app.config.get("UPLOAD_FOLDER", "uploads"))
            upload_dir.mkdir(parents=True, exist_ok=True)

            original_filename = file.filename
            tmp_id = str(uuid.uuid4())
            saved_name = f"{tmp_id}_{original_filename}"
            saved_path = upload_dir / saved_name
            file.save(saved_path)

            # Normalisation du metadata
            file_id = file_metadata.get("file_id") or tmp_id
            # Optionnel : tu peux override file_path pour refléter l'emplacement réel
            file_metadata["file_path"] = str(saved_path)
            file_metadata["file_id"] = file_id

            # 1. Chargement des documents depuis le fichier
            docs = self.ingester.load(file_metadata)
            if not docs:
                raise RuntimeError("Aucun document généré par l'ingestion.")

            # 2. Préparation des inputs pour upsert : textes, métadatas, ids
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            # IDs uniques : tu peux utiliser l'ID d'origine + index
            ids = [f"{doc.metadata.get('id', str(uuid.uuid4()))}_{i}" for i, doc in enumerate(docs)]

            # 3. Upsert dans la collection (nom basée sur file_id)
            collection_name = file_id  # tu peux appliquer une transformation si nécessaire
            self.vector_store.upsert_documents(
                collection_name=collection_name,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )

            self.logger.info(f"Ingestion réussie pour {collection_name}, {len(texts)} documents.")

            return {
                "status": "success",
                "collection": collection_name,
                "documents_indexed": len(texts),
                "file_id": file_id,
            }

        except Exception as e:
            self.logger.error(f"IngestService.ingest_file failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}, 500

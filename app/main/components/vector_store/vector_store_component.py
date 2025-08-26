import chromadb
from chromadb.config import Settings
from loguru import logger
from flask import current_app
from app.main.components.embedding.embedding_component import EmbeddingComponent

class VectorStoreComponent:
    _instance = None

    def __new__(cls, embedding_component: EmbeddingComponent):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logger
            cls._instance.embedding_component = embedding_component
            cls._instance.client = None  # initialisation différée
        return cls._instance

    def _ensure_client(self):
        if self.client is None:
            # ici on est censé être dans un contexte Flask valide
            persist_directory = current_app.config.get("DB_VECTOR_FOLDER", "db")
            self.logger.info(f"Initializing Chroma client at {persist_directory}")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings()
            )

    def get_collection(self, name: str):
        self._ensure_client()
        return self.client.get_or_create_collection(name=name)

    def upsert_documents(
        self,
        collection_name: str,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ):
        col = self.get_collection(collection_name)
        embeddings = self.embedding_component.embed(texts)
        col.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )
        self.logger.info(f"Upserted {len(texts)} docs into collection '{collection_name}'.")

    def get_retriever(self, collection_name: str, k: int, threshold: float):
        col = self.get_collection(collection_name)
        return col.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": threshold},
        )

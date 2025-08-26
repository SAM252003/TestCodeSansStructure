import os
from typing import List, Dict, Optional, Set

from langchain.schema import Document
from loguru import logger

from app.main.components.ingest.ingest_helper import (
    get_docs_pdf,
)


class IngestComponent:
    """
    Composant bas niveau qui sait charger un fichier (PDF, CSV, etc.)
    et le transformer en une liste de Documents LangChain.
    """

    ALLOWED_EXTENSIONS: Set[str] = {
        "pdf",
    }

    MAPPER: Dict[str, any] = {
        "pdf": get_docs_pdf,
    }

    def __init__(self) -> None:
        self.logger = logger
        self.logger.info("IngestComponent instance created.")

    def _get_extension(self, filename: str) -> Optional[str]:
        if "." not in filename:
            return None
        ext = filename.rsplit(".", 1)[-1].lower()
        return ext if ext in self.ALLOWED_EXTENSIONS else None

    def load(self, file_metadata: Dict) -> Optional[List[Document]]:
        try:
            self.logger.info(f"Loading file for ingestion: {file_metadata}")
            file_path = file_metadata.get("file_path")
            if not file_path:
                raise ValueError("file_path missing in metadata")
            file_id = file_metadata.get("file_id")

            file_name = os.path.basename(file_path)
            ext = self._get_extension(file_name)
            if not ext:
                self.logger.error("Unsupported file type: {}", file_name)
                raise ValueError(f"Unsupported file type: {file_name}")

            loader = self.MAPPER.get(ext)
            if loader is None:
                self.logger.error("No loader for extension: {}", ext)
                raise ValueError(f"No loader for extension {ext}")

            docs: List[Document] = loader(file_path, file_id)
            self.logger.info(f"Created {len(docs)} document(s) from {file_name}")
            return docs
        except Exception as e:
            self.logger.error(f"IngestComponent.load failed: {e}", exc_info=True)
            raise

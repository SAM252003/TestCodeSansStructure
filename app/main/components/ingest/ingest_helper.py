from typing import List, Optional
from langchain.schema import Document
import uuid
import os

def _make_dummy_doc(text: str, source: str):
    return Document(
        page_content=text,
        metadata={
            "section": "dummy",
            "source": source,
            "id": str(uuid.uuid4()),
        },
    )

def get_docs_pdf(file_path: str, file_id: Optional[str] = None) -> List[Document]:
    # TODO: remplacer par un vrai parse PDF
    # Pour test, on lit tout en texte brut
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        content = f"Impossible de lire {file_path} (stub)."
    return [_make_dummy_doc(content[:1000], os.path.basename(file_path))]

from pathlib import Path
import re, uuid
from typing import List
from langchain.schema import Document
from markitdown import MarkItDown
md_converter = MarkItDown(enable_plugins=False)

def load_livre_dor(pdf_path: str | Path) -> List[Document]:
    pdf_path = Path(pdf_path)
    md_text  = md_converter.convert(pdf_path).text_content
    blocks   = re.split(r'\n(?=# )', md_text)
    docs = []
    for blk in blocks:
        if not blk.strip():
            continue
        title = blk.splitlines()[0].lstrip("# ").strip()
        docs.append(
            Document(
                page_content=blk,
                metadata={
                    "section": title,
                    "source": pdf_path.name,
                    "id": str(uuid.uuid4())
                }
            )
        )
    return docs

# ingest/extract_images.py

import fitz            # pip install pymupdf
from pathlib import Path
from google.cloud import vision
from PIL import Image
import io

# Initialisation du client (clé prise depuis GOOGLE_APPLICATION_CREDENTIALS)
client = vision.ImageAnnotatorClient()

def extract_images(pdf_path: str | Path, out_dir: str = "data/images"):
    pdf = fitz.open(pdf_path)
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)
    images = []
    for pageno in range(len(pdf)):
        for img_index, img in enumerate(pdf.get_page_images(pageno)):
            xref = img[0]
            pix = pdf.extract_image(xref)
            img_bytes, ext = pix["image"], pix["ext"]
            img_path = out / f"page{pageno+1}_{img_index+1}.{ext}"
            img_path.write_bytes(img_bytes)
            images.append({"path": str(img_path), "page": pageno+1})
    return images

def ocr_image_cloud(path: str) -> str:
    """OCR via Google Cloud Vision API"""
    with io.open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")
    # full_text_annotation contient tout le texte détecté
    return response.full_text_annotation.text or ""


from pypdf import PdfReader
import docx2txt
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract



def ocr_pdf(file_path: str) -> str:
    """
    Extrait le texte d'un PDF scanné (image) via OCR.
    Rasterise chaque page en image, puis passe l'image à Tesseract.
    """
    text = ""
    pages = convert_from_path(file_path, dpi=300)

    for page_image in pages:
        text += pytesseract.image_to_string(page_image, lang="fra")
        text += "\n"

    return text


# Extrait le texte d'une pièce jointe 
def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        # Si l'extraction normale n'a presque rien donné → PDF scanné → OCR
        if len(text.strip()) < 20:
            text = ocr_pdf(file_path)

        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")

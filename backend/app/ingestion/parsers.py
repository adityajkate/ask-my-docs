from io import BytesIO
from pathlib import Path
from typing import BinaryIO
import re


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_text_file(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".html", ".htm"}:
        return normalize_text(data.decode("utf-8", errors="ignore"))
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_text("\n\n".join(pages))
    if suffix == ".docx":
        from docx import Document

        doc = Document(BytesIO(data))
        return normalize_text("\n".join(paragraph.text for paragraph in doc.paragraphs))
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def parse_upload(filename: str, file: BinaryIO) -> str:
    return parse_text_file(filename, file.read())


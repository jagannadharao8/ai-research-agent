from pypdf import PdfReader
import os


def extract_text_from_pdf(pdf_path):
    """
    Extract raw text from PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found.")

    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text


def split_text_into_chunks(text, chunk_size=500):
    """
    Split text into smaller chunks for better embedding.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end

    return chunks


def load_pdf_as_documents(pdf_path):
    """
    Returns structured documents from PDF.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(raw_text)

    documents = []

    for chunk in chunks:
        documents.append({
            "source": "pdf",
            "content": chunk
        })

    return documents
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


def split_text_into_chunks(text, chunk_size=500, overlap=100):
    """
    Split text into smaller chunks for better embedding.
    Uses word boundaries and overlap to preserve semantics.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            
            # Keep words for overlap
            overlap_chunk = []
            overlap_length = 0
            for w in reversed(current_chunk):
                overlap_chunk.insert(0, w)
                overlap_length += len(w) + 1
                if overlap_length >= overlap:
                    break
            current_chunk = overlap_chunk
            current_length = overlap_length
            
    if current_chunk:
        joined_chunk = " ".join(current_chunk)
        if not chunks or joined_chunk != chunks[-1]:
            chunks.append(joined_chunk)
        
    return chunks


def load_pdf_as_documents(pdf_path):
    """
    Returns structured documents from PDF.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(raw_text)
    filename = os.path.basename(pdf_path)

    documents = []

    for chunk in chunks:
        documents.append({
            "source": "pdf",
            "title": filename,
            "url": "",
            "content": chunk
        })

    return documents
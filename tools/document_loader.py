import os
import base64
from pypdf import PdfReader
from docx import Document
from groq import Groq

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = ""
    for para in doc.paragraphs:
        full_text += para.text + "\n"
    return full_text

def extract_text_from_image(file_path):
    """Uses Groq Vision to transcribe an image."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
            
    if not api_key:
        return "GROQ API KEY not found for image processing."
        
    client = Groq(api_key=api_key)
    
    with open(file_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    prompt = "Extract and transcribe all text from this image exactly as written. If there are charts or diagrams, describe the data and insights they convey in detail."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Image processing failed: {e}"


def split_text_into_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            
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

def load_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'pdf':
        raw_text = extract_text_from_pdf(file_path)
        source_type = "pdf"
    elif ext in ['doc', 'docx']:
        raw_text = extract_text_from_docx(file_path)
        source_type = "word"
    elif ext in ['png', 'jpg', 'jpeg']:
        raw_text = extract_text_from_image(file_path)
        source_type = "image"
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    chunks = split_text_into_chunks(raw_text)
    filename = os.path.basename(file_path)
    documents = []

    for chunk in chunks:
        documents.append({
            "source": source_type,
            "title": filename,
            "url": "",
            "content": chunk
        })

    return documents

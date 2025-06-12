import re
import nltk
import numpy as np
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

def load_document(file_path: str) -> str:
    """Load document with format-specific parsing"""
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format.")
            
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
    return text.strip()

def semantic_chunking(text: str, model, max_chars=500, min_chars=100, threshold=0.85) -> list[str]:
    """
    Split text into semantic chunks using embedding similarity
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    
    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        if not current_chunk:
            current_chunk.append(sentence)
            current_size += len(sentence)
            continue
        
        if i > 0:
            similarity = np.dot(embeddings[i], embeddings[i-1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
            )
            
            if similarity < threshold or current_size + len(sentence) > max_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        current_chunk.append(sentence)
        current_size += len(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "(?<=\. )", " "],
                chunk_size=max_chars,
                chunk_overlap=min(50, max_chars // 10)
            )
            final_chunks.extend(splitter.split_text(chunk))
        elif len(chunk) >= min_chars:
            final_chunks.append(chunk)
    
    return final_chunks

def chunk_text(text: str, model_name: str) -> list[str]:
    """Hybrid chunking with semantic segmentation"""
    sections = re.split(r'\n{2,}', text)
    
    model = SentenceTransformer(model_name)
    chunks = []
    for section in sections:
        chunks.extend(semantic_chunking(section, model))
    
    return chunks

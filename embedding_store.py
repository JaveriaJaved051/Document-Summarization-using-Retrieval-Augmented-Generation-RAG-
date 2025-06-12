import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        
    def create_index(self, chunks: list[str]):
        self.chunks = chunks
        embeddings = self.embedder.encode(chunks, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
    def save_index(self, file_path: str):
        faiss.write_index(self.index, file_path)
        
    def load_index(self, file_path: str):
        self.index = faiss.read_index(file_path)
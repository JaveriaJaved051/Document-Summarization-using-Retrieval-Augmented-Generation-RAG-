from langchain.vectorstores.base import VectorStore

def retrieve_chunks(
    query: str, 
    vector_store: VectorStore, 
    top_k: int
) -> tuple[list[str], list[float]]:
    query_embed = vector_store.embedder.encode([query])
    distances, indices = vector_store.index.search(query_embed, top_k)
    
    retrieved_chunks = [vector_store.chunks[i] for i in indices[0]]
    similarity_scores = [1 / (1 + d) for d in distances[0]]  # Convert distance to similarity
    
    return retrieved_chunks, similarity_scores
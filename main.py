from config import *
from document_loader import load_document, chunk_text
from embedding_store import VectorStore
from retriever import retrieve_chunks
# from summarizer import Summarizer
from summarizer import Llama3Summarizer

import time
from summarizer import Llama3Summarizer 

def summarize_document(file_path: str):
    # 1. Load and chunk document with semantic segmentation
    start_time = time.time()
    text = load_document(file_path)
    chunks = chunk_text(text, EMBEDDING_MODEL)  # Pass embedding model name
    load_time = time.time() - start_time
    
    # 2. Create embeddings
    start_embed = time.time()
    vector_store = VectorStore(EMBEDDING_MODEL)
    vector_store.create_index(chunks)
    embed_time = time.time() - start_embed
    
    # 3. Retrieve relevant chunks
    start_retrieve = time.time()
    retrieved_chunks, similarity_scores = retrieve_chunks(
        QUERY, vector_store, RETRIEVAL_TOP_K
    )
    retrieve_time = time.time() - start_retrieve
    
    
    # 4. Generate summary with LLaMA 3
    start_summarize = time.time()
    context = "\n\n".join([
        f"CONTEXT SEGMENT {i+1} (Relevance: {score:.2f}):\n{chunk}"
        for i, (chunk, score) in enumerate(zip(retrieved_chunks, similarity_scores))
    ])
    
    summarizer = Llama3Summarizer(SUMMARIZATION_MODEL)
    summary = summarizer.generate_summary(context)
    summarize_time = time.time() - start_summarize
    
    # 5. Prepare metrics
    total_time = time.time() - start_time
    metrics = {
        "load_time": f"{load_time:.2f}s",
        "embed_time": f"{embed_time:.2f}s",
        "retrieve_time": f"{retrieve_time:.2f}s",
        "summarize_time": f"{summarize_time:.2f}s",
        "total_time": f"{total_time:.2f}s",
        "chunk_count": len(chunks),
        "retrieved_chunks": RETRIEVAL_TOP_K
    }
    
    return {
        "retrieved_chunks": retrieved_chunks,
        "similarity_scores": similarity_scores,
        "summary": summary,
        "metrics": metrics
    }
if __name__ == "__main__":
    file_path = "sample_documents/RAG.pdf"  # ✅ Replace with your actual file name if different

    result = summarize_document(file_path)

    print("\n📄 --- Summary ---\n")
    print(result["summary"])

    print("\n📌 --- Retrieved Chunks ---\n")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"Chunk {i}:\n{chunk}\n")

    print("\n⏱️ --- Metrics ---\n")
    for key, value in result["metrics"].items():
        print(f"{key}: {value}")
    print("\n✅ Summary generation completed successfully!")
    print("\n--- End of Summary ---\n")
    print("Thank you for using the summarization tool!")
    
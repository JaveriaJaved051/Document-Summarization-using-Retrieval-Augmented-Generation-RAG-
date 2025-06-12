from main import summarize_document
import json

def run_examples():
    documents = [
        "sample_documents/RAG.pdf",
        "sample_documents/example.txt",
        "sample_documents/example.md"
    ]
    
    results = {}
    for doc_path in documents:
        print(f"Processing {doc_path}...")
        results[doc_path] = summarize_document(doc_path)
    
    # Save results
    with open("sample_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Sample results saved to sample_results.json")

if __name__ == "__main__":
    run_examples()
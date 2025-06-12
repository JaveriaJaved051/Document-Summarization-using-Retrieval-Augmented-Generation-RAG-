# 📄 RAG Document Summarization System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that summarizes large documents by first retrieving semantically relevant content and then generating fluent summaries using Large Language Models (LLMs).

---

## 🎯 Objective

To develop a modular summarization pipeline that:
- Accepts large documents (PDF, TXT, Markdown)
- Splits them into semantically meaningful chunks
- Embeds them using transformer-based models
- Retrieves top relevant chunks via semantic search
- Generates a high-quality summary using LLMs

---

## ⚙️ Setup Instructions

###  Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate rag-summarization



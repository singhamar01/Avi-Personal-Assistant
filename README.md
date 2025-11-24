# 🤖 Avi - Local Personal Data Assistant

Avi is a secure, offline Personal Assistant designed to answer questions based on your private documents. It runs entirely on your local machine using lightweight Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

**Privacy First:** No data leaves your computer. Your documents and the AI processing remain 100% offline.

## 🚀 Features
- **Offline Capability:** Runs locally on Windows/Mac without internet dependency.
- **Document Ingestion:** automatically reads and indexes PDF files from a dedicated folder.
- **Contextual Awareness:** Uses Vector Embeddings (ChromaDB) to find relevant info before answering.
- **Model Flexibility:** Switch between speed (`llama3.2:1b`) and accuracy (`llama3.1:8b`).
- **User Interface:** Clean, browser-based chat interface using Streamlit.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **LLM Engine:** [Ollama](https://ollama.com/)
- **Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Frontend:** Streamlit

## 📋 Prerequisites
1. **Python:** Ensure Python is installed.
2. **Ollama:** Download and install Ollama from [ollama.com](https://ollama.com).
3. **Models:** Run the following commands in your terminal to pull the required models:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
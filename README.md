# 🧠 Enterprise Knowledge Base (Local RAG)

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system that allows users to chat with their PDFs and text documents. Built completely from scratch using LangChain, FAISS, Hugging Face, and Streamlit.

## 💡 The Problem
Enterprise knowledge is trapped in PDFs and documents. Cloud LLMs (like ChatGPT) are expensive and raise severe data privacy issues. This project solves that by running entirely on local hardware.

## 🏗️ Architecture
1. **Ingestion (`ingest.py`):** Loads PDFs, splits them into chunks, and creates vector embeddings using `all-MiniLM-L6-v2`.
2. **Storage:** Vectors are stored locally in a **FAISS** index.
3. **Retrieval:** When a user asks a question, the system searches the FAISS index for the most relevant chunks.
4. **Generation (`streamlit_app.py`):** The retrieved chunks and the user's question are fed into a local **Qwen2.5-0.5B-Instruct** model to generate a fact-based answer.
5. **UI:** A ChatGPT-style web interface built with Streamlit.

## 🚀 How to Run Locally
1. Clone the repo: `git clone https://github.com/ankithavarikollu/enterprise-rag-knowledge-base.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `.\venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Add your PDFs to the `data/` folder.
6. Build the index: `python ingest.py`
7. Launch the app: `streamlit run streamlit_app.py`

## 🛠️ Tech Stack
* **Python**
* **LangChain** (Orchestration)
* **FAISS** (Vector Database)
* **Hugging Face Transformers** (Local LLMs & Embeddings)
* **Streamlit** (Web UI)
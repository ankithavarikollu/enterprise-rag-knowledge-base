import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        if filename.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif filename.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs

def main():
    print("Loading documents from data/ ...")
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s).")

    print("Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("Building embeddings (first run downloads the model, ~90 MB) ...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building FAISS index ...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"Done. Index saved to ./{INDEX_DIR}/")

if __name__ == "__main__":
    main()
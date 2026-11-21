import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def build_chain():
    # 1. Load the exact same embedding model used in ingest.py
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Load your local FAISS index
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Choose the LLM
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("Using OpenAI (gpt-4o-mini)")
    else:
        from transformers import pipeline
        from langchain_community.llms import HuggingFacePipeline
        print("No OPENAI_API_KEY set — using local flan-t5-base (first run downloads ~1 GB)")
        pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=pipe)

    # 4. Create the RAG chain
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=retriever, return_source_documents=True,
    )

if __name__ == "__main__":
    print("Loading models and index...")
    chain = build_chain()
    
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        result = chain.invoke({"query": q})
        print("\nAnswer:", result["result"])
        print("\nSources:")
        for i, d in enumerate(result["source_documents"], 1):
            print(f"  [{i}] {d.metadata.get('source', 'unknown')}")
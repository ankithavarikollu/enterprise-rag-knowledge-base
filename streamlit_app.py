import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# --- PAGE SETUP ---
st.set_page_config(page_title="Enterprise Knowledge Base", page_icon="🧠")
st.title("🧠 Enterprise Knowledge Base")
st.markdown("Ask questions based on the documents in your `data` folder.")

# --- LOAD MODELS (Cached so it doesn't reload every time you type) ---
@st.cache_resource(show_spinner="Loading AI models and index...")
def load_rag_chain():
    # 1. Load Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Load Local LLM (using Qwen for better answers)
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    # 3. Create a clean prompt for the small local model
    prompt_template = """Use the following context to answer the question. 
If you don't know the answer based on the context, just say "I don't know."

Context:
{context}

Question: {question}

Helpful Answer:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 4. Return the Chain with the new prompt
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

chain = load_rag_chain()

# --- CHAT INTERFACE ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = chain.invoke({"query": prompt})
            full_text = result["result"]
            
            # Clean up the answer (only show what's after "Helpful Answer:")
            if "Helpful Answer:" in full_text:
                clean_answer = full_text.split("Helpful Answer:")[-1].strip()
            else:
                clean_answer = full_text.strip()
            
            # Display answer
            st.markdown(clean_answer)
            
            # Display sources in an expander
            with st.expander("View Sources"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.write(f"**Source {i}:** `{doc.metadata.get('source', 'unknown')}`")
                    st.info(doc.page_content[:300] + "...")

    # Save assistant response to history (save the clean answer, not the full messy text)
    st.session_state.messages.append({"role": "assistant", "content": clean_answer})
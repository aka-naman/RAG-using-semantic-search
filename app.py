import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.ingestion import load_documents, chunk_documents
from src.indexing import HybridIndex
from src.retrieval import HybridRetriever
from src.generation import OpenAIRAGGenerator, LocalRAGGenerator

# 1. Setup
load_dotenv()
st.set_page_config(page_title="RAG Explorer", layout="wide")

# 2. Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 3. Sidebar Configuration
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    
    # Model Selection
    mode = st.radio("Execution Mode", ["Offline (Local LLM)", "Online (OpenAI)"])
    model_id = st.text_input("Model ID", 
                            value="unsloth/Llama-3.2-1B-Instruct" if mode == "Offline (Local LLM)" else "gpt-4o")
    
    # RAG Settings
    st.subheader("Retrieval Settings")
    chunk_size = st.slider("Chunk Size", 200, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
    top_k = st.number_input("Top K Context", 1, 10, 5)
    
    # Reset Index Button
    if st.button("🔄 Rebuild Index"):
        with st.spinner("Indexing documents (this may take a minute)..."):
            doc_dir = os.getenv("DOCUMENT_DIR", "./documents")
            index_dir = os.getenv("INDEX_DIR", "./index")
            embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            
            # Clear existing cache for retriever
            st.cache_resource.clear()
            
            docs = load_documents(doc_dir)
            if docs:
                chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                index = HybridIndex(embedding_model_name=embedding_model)
                index.create_bm25_index(chunks)
                index.create_vector_index(chunks)
                index.save_index(index_dir)
                st.success(f"Successfully indexed {len(docs)} files ({len(chunks)} chunks)!")
            else:
                st.error("No documents found in 'documents/' directory.")

    # Show indexed files
    st.subheader("📂 Indexed Documents")
    doc_dir = os.getenv("DOCUMENT_DIR", "./documents")
    if os.path.exists(doc_dir):
        files = os.listdir(doc_dir)
        if files:
            for f in files:
                st.text(f"📄 {f}")
        else:
            st.text("No files found.")

# 4. Resource Caching
@st.cache_resource
def get_retriever():
    index_dir = os.getenv("INDEX_DIR", "./index")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    if os.path.exists(os.path.join(index_dir, "bm25_index.pkl")):
        index = HybridIndex.load_index(index_dir, embedding_model_name=embedding_model)
        return HybridRetriever(index)
    return None

@st.cache_resource
def get_generator(mode, model_id):
    if "Offline" in mode:
        return LocalRAGGenerator(model_id=model_id)
    else:
        return OpenAIRAGGenerator(model_name=model_id)

# 5. Main UI
st.title("📚 Ask My Docs - Dynamic RAG")

# File Upload Section
uploaded_files = st.file_uploader("Upload new documents (PDF, DOCX, CSV, etc.)", accept_multiple_files=True)
if uploaded_files:
    doc_dir = os.getenv("DOCUMENT_DIR", "./documents")
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
    
    for uploaded_file in uploaded_files:
        with open(os.path.join(doc_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.info("Files uploaded! Click 'Rebuild Index' in the sidebar to process them.")

# 6. Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Generate response
    retriever = get_retriever()
    if not retriever:
        st.error("Index not found. Please click 'Rebuild Index' in the sidebar.")
    else:
        generator = get_generator(mode, model_id)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                context = retriever.retrieve(prompt, top_k=top_k)
                
                # Show context in expander
                with st.expander("🔍 View Retrieved Context & Sources"):
                    for i, chunk in enumerate(context):
                        source = chunk.metadata.get('source', 'Unknown')
                        method = chunk.metadata.get('method', 'N/A')
                        st.write(f"**Chunk {i+1}** | Source: `{source}` | Method: `{method}`")
                        st.info(chunk.content)
                
                answer = generator.generate_answer(prompt, context)
                st.markdown(answer)
                
                # Evaluation (Optional)
                if mode == "Online (OpenAI)":
                    eval_res = generator.self_correct(prompt, context, answer)
                    st.caption(f"Self-Evaluation: {eval_res['evaluation']}")
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

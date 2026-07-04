import os
import uuid
import shutil
import streamlit as st
from pinecone import Pinecone
from ingest import ingest_documents
from rag_chain import ask_question
from config import PINECONE_API_KEY, PINECONE_INDEX

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Stylesheet for UI Redesign
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Main font override */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Custom premium container card */
.feature-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* File badge list styling */
.file-badge {
    background: rgba(59, 130, 246, 0.1);
    color: #60a5fa;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    border: 1px solid rgba(59, 130, 246, 0.2);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

/* Premium gradient titles */
.gradient-title {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.5rem;
    margin-bottom: 0.25rem;
}

/* Custom chatbot bubble tweaks */
.stChatMessage {
    padding: 1.25rem !important;
    border-radius: 16px !important;
    margin-bottom: 1.25rem !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
}

/* User Message specific card style */
.stChatMessage[data-testid="stChatMessageUser"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.15) !important;
}

/* Assistant Message specific card style */
.stChatMessage[data-testid="stChatMessageAssistant"] {
    background: linear-gradient(135deg, #181d2a 0%, #111420 100%) !important;
    border: 1px solid rgba(139, 92, 246, 0.15) !important;
}

/* Styled primary button override */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
}

/* Sub-headings */
.sidebar-header {
    color: #94a3b8;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE INITIALIZATION ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:12]}"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_prepared" not in st.session_state:
    st.session_state.db_prepared = False

# ---------------- SIDEBAR (FILE UPLOADER & CONFIGS) ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">🛠️ Model Settings</div>', unsafe_allow_html=True)
    
    # Model Selector
    model_opt = st.selectbox(
        "Select Answering Model:",
        options=["llama-3.1-8b-instant (Fast)", "llama-3.3-70b-versatile (Accurate)"],
        index=0,
        label_visibility="collapsed"
    )
    model_name = "llama-3.1-8b-instant" if "Fast" in model_opt else "llama-3.3-70b-versatile"
    
    st.markdown('<div class="sidebar-header">📁 Upload Documents</div>', unsafe_allow_html=True)
    
    # Document uploader (limit up to 5 files of varying extensions)
    uploaded_files = st.file_uploader(
        "Drag & Drop or Browse Files",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md"],
        accept_multiple_files=True,
        help="Upload up to 5 documents to start searching"
    )

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("⚠️ Max limit is 5 files. Only the first 5 files will be processed.")
            uploaded_files = uploaded_files[:5]

    # Convert currently uploaded files to fingerprints (name + size) for change detection
    current_file_fingerprints = [(f.name, f.size) for f in uploaded_files] if uploaded_files else []
    
    # Detect file additions/deletions
    if current_file_fingerprints != st.session_state.uploaded_files:
        st.session_state.uploaded_files = current_file_fingerprints
        
        if uploaded_files:
            # Create a clean temp uploads directory
            temp_dir = os.path.join(os.getcwd(), "temp_uploads")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            temp_paths = []
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                temp_paths.append(temp_path)
            
            # Ingest all files into Pinecone
            with st.spinner("Processing & Ingesting documents... ⏳"):
                try:
                    ingest_documents(temp_paths, st.session_state.session_id)
                    st.session_state.db_prepared = True
                    st.sidebar.success("✅ Knowledge base built!")
                except Exception as e:
                    st.sidebar.error(f"❌ Ingestion failed: {str(e)}")
                    st.session_state.db_prepared = False
            
            # Clean up temp folder
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        else:
            # All documents removed: clean up Pinecone vectors for this session
            if st.session_state.db_prepared:
                with st.spinner("Cleaning up database..."):
                    try:
                        pc = Pinecone(api_key=PINECONE_API_KEY)
                        index = pc.Index(PINECONE_INDEX)
                        index.delete(delete_all=True, namespace=st.session_state.session_id)
                    except Exception as e:
                        print(f"Error purging Pinecone namespace: {e}")
            st.session_state.db_prepared = False
            st.sidebar.info("Knowledge base cleared.")

    # Show Active Files Badge List
    if uploaded_files:
        st.markdown('<div class="sidebar-header">📂 Active Documents</div>', unsafe_allow_html=True)
        for f in uploaded_files:
            size_mb = f.size / (1024 * 1024)
            st.markdown(f"""
            <div class="file-badge">
                <span>🗎 <b>{f.name}</b></span>
                <span style="font-size: 0.75rem; opacity: 0.8;">{size_mb:.2f} MB</span>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Reset/Clear Session Button
    if st.button("🗑️ Clear Chat & Reset Session"):
        # Purge Pinecone namespace
        if st.session_state.db_prepared:
            with st.spinner("Wiping database namespace..."):
                try:
                    pc = Pinecone(api_key=PINECONE_API_KEY)
                    index = pc.Index(PINECONE_INDEX)
                    index.delete(delete_all=True, namespace=st.session_state.session_id)
                except Exception as e:
                    print(f"Error wiping Pinecone namespace: {e}")
                    
        # Reset session state values
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:12]}"
        st.session_state.uploaded_files = []
        st.session_state.messages = []
        st.session_state.db_prepared = False
        st.rerun()

# ---------------- MAIN CHAT PANEL ----------------
# Title Layout
st.markdown('<div class="gradient-title">⚡ Multi-Document AI Assistant</div>', unsafe_allow_html=True)
st.write("Perform semantic search and query answering across multiple uploaded document formats simultaneously.")
st.markdown("---")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Block
if not st.session_state.db_prepared:
    st.info("👈 Please upload one or more files (PDF, Word, Excel, CSV, Text) in the sidebar to start asking questions.")
    user_prompt = st.chat_input("Chat disabled. Upload files to proceed...", disabled=True)
else:
    user_prompt = st.chat_input("Ask a question about your uploaded documents...")

if user_prompt:
    # Display user prompt in chat
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Process & display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents & generating answer... 🤖"):
            try:
                response = ask_question(
                    query=user_prompt,
                    namespace=st.session_state.session_id,
                    model_name=model_name
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                err_msg = f"⚠️ An error occurred while answering: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

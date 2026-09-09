import streamlit as st
import os
from dotenv import load_dotenv
from rag_utility import process_document_to_chroma_db, answer_question

# Load environment variables from .env file
load_dotenv()

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="Groq RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ------------------------
# Custom CSS Styling
# ------------------------
st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .hero {
        background: linear-gradient(135deg, #FF6B6B 0%, #C2185B 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(194, 24, 91, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #FFE3E9;
        font-size: 1rem;
        margin: 0;
    }

    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.2rem;
        border: 1px solid #eef0f7;
    }

    .status-badge-ready {
        display: inline-block;
        background: #E6F9F0;
        color: #12805C;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid #B7EEDA;
        margin-bottom: 0.5rem;
    }

    .status-badge-waiting {
        display: inline-block;
        background: #FFF4E5;
        color: #B7791F;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid #FBD38D;
        margin-bottom: 0.5rem;
    }

    .answer-box {
        background: #F8F5FF;
        border-left: 4px solid #8B5CF6;
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        margin-top: 0.8rem;
        line-height: 1.6;
    }

    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #C2185B 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }

    .stTextInput>div>div>input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Hero Header
# ------------------------
st.markdown("""
<div class="hero">
    <h1>🤖 Groq-Powered PDF Chatbot</h1>
    <p>Upload a PDF and ask questions based on its content — answered instantly by Groq</p>
</div>
""", unsafe_allow_html=True)

# ------------------------
# API Key Check
# ------------------------
if not os.getenv("GROQ_API_KEY"):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.error("⚠️ Missing GROQ_API_KEY. Add it to your `.env` file as: `GROQ_API_KEY=your_key_here`")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if "processed" not in st.session_state:
    st.session_state.processed = False

# ------------------------
# Upload & Process Section
# ------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("#### 📄 Step 1 — Upload your document")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf", label_visibility="collapsed")

if uploaded_file:
    with open("temp_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown(f"**File:** `{uploaded_file.name}`")

    if st.session_state.processed:
        st.markdown('<span class="status-badge-ready">✅ Ready to chat</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge-waiting">⏳ Not analyzed yet</span>', unsafe_allow_html=True)

    if st.button("🔍 Analyze PDF"):
        with st.spinner("Processing document..."):
            try:
                process_document_to_chroma_db("temp_file.pdf")
                st.session_state.processed = True
                st.success("Analysis complete! You can now ask questions below.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------
# Q&A Section
# ------------------------
if st.session_state.processed:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 💬 Step 2 — Ask a question")

    user_input = st.text_input(
        "Ask a question about the PDF",
        placeholder="e.g., What is the main conclusion of this document?",
        label_visibility="collapsed"
    )

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = answer_question(user_input)
                st.markdown(f"""
                <div class="answer-box">
                    {response}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

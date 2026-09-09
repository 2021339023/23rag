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
# Custom CSS Styling — Dark Theme
# ------------------------
st.markdown("""
<style>
    .stApp, .main, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0E1117 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .hero {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.8rem;
        border: 1px solid #2D3748;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.15);
    }
    .hero h1 {
        color: #F3F4F6;
        font-size: 2rem;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #9CA3AF;
        font-size: 1rem;
        margin: 0;
    }
    .hero .accent {
        background: linear-gradient(90deg, #A78BFA, #22D3EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-card, .section-card * {
        color: #E5E7EB !important;
    }
    .section-card {
        background: #161B22;
        padding: 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        border: 1px solid #2D3748;
    }
    .section-card h4 {
        color: #F9FAFB !important;
        font-weight: 600;
    }

    .status-badge-ready {
        display: inline-block;
        background: rgba(34, 197, 94, 0.12);
        color: #4ADE80 !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(74, 222, 128, 0.35);
        margin-bottom: 0.5rem;
    }

    .status-badge-waiting {
        display: inline-block;
        background: rgba(251, 191, 36, 0.12);
        color: #FBBF24 !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(251, 191, 36, 0.35);
        margin-bottom: 0.5rem;
    }

    .answer-box, .answer-box * {
        color: #E5E7EB !important;
    }
    .answer-box {
        background: #1A1F2B;
        border-left: 4px solid #A78BFA;
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        margin-top: 0.8rem;
        line-height: 1.7;
        overflow-x: auto;
    }
    .answer-box table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.6rem;
    }
    .answer-box th, .answer-box td {
        padding: 8px 12px;
        border: 1px solid #2D3748;
        text-align: left;
        background: transparent;
    }
    .answer-box th {
        background: #232A3B !important;
        font-weight: 600;
        color: #C4B5FD !important;
    }
    .answer-box tr:nth-child(even) td {
        background: #1D2333;
    }

    .stButton>button {
        background: linear-gradient(135deg, #7C3AED 0%, #22D3EE 100%);
        color: #0E1117;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        border: none;
        width: 100%;
        transition: opacity 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.85;
    }

    .stTextInput>div>div>input, .stFileUploader {
        border-radius: 10px;
    }

    .stTextInput>div>div>input {
        background-color: #0E1117 !important;
        color: #F3F4F6 !important;
        border: 1px solid #2D3748 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #0E1117 !important;
        border: 1px dashed #3B4252 !important;
    }

    ::placeholder {
        color: #6B7280 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Hero Header
# ------------------------
st.markdown("""
<div class="hero">
    <h1>🤖 <span class="accent">Groq-Powered</span> PDF Chatbot</h1>
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

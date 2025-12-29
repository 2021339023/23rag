import streamlit as st
import os
from dotenv import load_dotenv
from rag_utility import process_document_to_chroma_db, answer_question

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Groq RAG Chatbot", page_icon="🤖")

st.title("🤖 Groq-Powered PDF Chatbot")
st.markdown("Upload a PDF document and ask questions based on its content.")

# Check if the API key is loaded into the environment
if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY. Ensure your .env file contains: GROQ_API_KEY=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")
    st.stop()

if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    with open("temp_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Analyze PDF"):
        with st.spinner("Processing document..."):
            try:
                process_document_to_chroma_db("temp_file.pdf")
                st.session_state.processed = True
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()

if st.session_state.processed:
    user_input = st.text_input("Ask a question about the PDF:")
    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = answer_question(user_input)
                st.subheader("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
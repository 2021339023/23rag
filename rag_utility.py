import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_document_to_chroma_db(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)

def answer_question(query: str) -> str:
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Using your provided key directly as a fallback if env is not found
    api_key = os.getenv("GROQ_API_KEY", "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")

    llm = ChatGroq(
        temperature=0,
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    system_prompt = (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]



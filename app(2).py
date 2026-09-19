import os
import streamlit as st
from dotenv import load_dotenv

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="PDF RAG with Groq", layout="wide")

st.title("📄 PDF RAG Application")
st.write("Upload a PDF, create embeddings using an open-source model, store them in FAISS, and ask questions using Groq LLM.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "uploaded_document.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Extracting PDF text..."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    with st.spinner("Creating chunks..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

    with st.spinner("Creating embeddings and storing in FAISS..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        st.session_state.vectorstore = vectorstore

    st.success(f"PDF processed successfully. Created {len(chunks)} chunks.")

question = st.text_input("Ask a question from your PDF")

if question and st.session_state.vectorstore:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("Please add GROQ_API_KEY in Streamlit secrets.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0
    )

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    with st.spinner("Generating answer..."):
        response = qa_chain.invoke({"query": question})

    st.subheader("Answer")
    st.write(response["result"])

    with st.expander("Sources"):
        for doc in response["source_documents"]:
            st.write(doc.page_content[:500])
            st.divider()

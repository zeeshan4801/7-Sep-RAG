import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(page_title="PDF RAG with Groq", layout="wide")

st.title("📄 PDF RAG Application")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])


if uploaded_file:

    pdf_path = "uploaded_document.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    documents = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    st.success(f"PDF processed. {len(chunks)} chunks created.")


question = st.text_input("Ask a question from your PDF")


if question:

    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("Add GROQ_API_KEY in Streamlit Secrets.")
        st.stop()


    llm = ChatGroq(
        groq_api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=0
    )


    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only this context:

        {context}

        Question:
        {question}
        """
    )


    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )


    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )


    response = chain.invoke(question)

    st.subheader("Answer")
    st.write(response.content)


    with st.expander("Sources"):
        for doc in retriever.invoke(question):
            st.write(doc.page_content[:500])
            st.divider()

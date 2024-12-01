

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader, PdfWriter
import os

# Title
st.title("AI-Based Generative Search System")
st.write("Upload multiple PDF documents to get precise answers to your queries!")

# Sidebar for Hugging Face API key
hf_api_key = st.sidebar.text_input("Enter your Hugging Face API Key:", type="password")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_api_key

# Step 1: Upload PDF Files
st.header("Step 1: Upload Your PDFs")
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write("Uploaded files:")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")

    # Step 2: Merge PDFs
    st.header("Step 2: Merging PDFs")
    merged_pdf_path = "merged_document.pdf"
    pdf_writer = PdfWriter()

    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    with open(merged_pdf_path, "wb") as f:
        pdf_writer.write(f)

    st.success(f"Merged PDF saved as {merged_pdf_path}")

    # Step 3: Load and Split Document
    st.header("Step 3: Load and Split Documents")
    loader = PyPDFLoader(merged_pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    splits = text_splitter.split_documents(pages)

    st.write(f"Number of document chunks created: {len(splits)}")

    # Step 4: Create Embeddings and Vector Store
    st.header("Step 4: Creating Vector Store")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_directory = './chroma_db'

    vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_directory)
    st.success(f"Vector store created with {vectordb._collection.count()} vectors.")

    # Step 5: Define Question and Query the Model
    st.header("Step 5: Ask Your Question")
    question = st.text_input("Enter your question:")

    if question:
        # Define the prompt template
        template = """
        You are a helpful assistant with access to detailed documents. Your task is to answer the question based on the provided context. If the context does not contain information relevant to the question, you should state that you don't know the answer rather than guessing.

        Use the following context to answer the question at the end. Provide a clear, concise, and accurate response. Your answer should be no longer than three sentences and always end with "Thanks for asking!"

        {context}

        Question:
        {question}

        Answer:
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Initialize the LLM
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.7})

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        # Get the answer
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": question})
        
        st.subheader("Answer:")
        st.write(result['result'])

        st.subheader("Source Document:")
        st.write(result['source_documents'][0]['page_content'])


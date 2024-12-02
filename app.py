import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub  # Updated
from langchain_community.vectorstores import Chroma  # Updated
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated
from langchain_community.document_loaders import PyPDFLoader  # Updated
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


# Set API Key for Hugging Face
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# Streamlit App Configuration
st.set_page_config(page_title="Insurance Chatbot", layout="wide")
st.title("Insurance Policy Query Chatbot")

# Sidebar for Uploading PDFs
st.sidebar.header("Upload Policy Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# Initialize Vector Store
@st.cache_resource
def initialize_vector_store(uploaded_files):
    if not uploaded_files:
        return None
    # Save PDFs locally and load them
    pdf_loader = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getvalue())
        pdf_loader.append(PyPDFLoader(file.name))
    
    # Merge all PDFs
    documents = []
    for loader in pdf_loader:
        documents.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # Initialize embedding and vector store
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./vector_store")
    return vector_store

if uploaded_files:
    vectordb = initialize_vector_store(uploaded_files)
    st.sidebar.success("Documents processed successfully!")
else:
    vectordb = None

# Chatbot Functionality
if vectordb:
    # Prompt Template
    template = """
    You are a helpful assistant with access to insurance policy documents.
    Answer the question based on the context below. If no relevant information is found in the context, 
    respond with "I don't know based on the provided documents."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Initialize the RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.7}),
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Chat Interface
    st.header("Chat with the Insurance Bot")
    user_input = st.text_input("Ask a question about the uploaded insurance policies:")
    if user_input:
        result = qa_chain({"query": user_input})
        st.markdown(f"### Answer:\n{result['result']}")
        
        # Display Source Document (Optional)
        with st.expander("Source Document"):
            source_doc = result["source_documents"][0]
            st.text(source_doc.page_content)
else:
    st.info("Please upload documents to enable the chatbot.")

# Additional Notes
st.sidebar.markdown("""
### Notes:
1. Ensure that the uploaded documents are relevant policy PDFs.
2. For questions outside the context of the documents, the bot will indicate it cannot provide an answer.
""")

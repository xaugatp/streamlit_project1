import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Set API Key for Hugging Face
api_key_filepath = "API_Key.txt"
with open(api_key_filepath, "r") as f:
    api_key = f.read().strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# Upload and Load PDF
st.title("AI-Based Generative Search System")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

    # Load and Split PDF
    loader = PyPDFLoader("uploaded.pdf")
    pages = loader.load()

    # Split PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    splits = text_splitter.split_documents(pages)
    st.write(f"Number of chunks created: {len(splits)}")

    # Create Embeddings and Vector Store using FAISS
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents=splits, embedding=embedding)

    # Define Prompt Template
    template = """
    Use the following context to answer the question. If the answer isn't in the context, say "I don't know."
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Ask a Question
    question = st.text_input("Enter your question:")
    if question:
        # Initialize LLM and Retrieval QA Chain
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        # Get and Display the Answer
        result = qa_chain({"query": question})
        st.write(f"Answer: {result['result']}")
        st.write("Source Document(s):")
        st.write(result["source_documents"][0])

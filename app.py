import sys
import solara
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

# Global State
uploaded_files = []
vectordb = None
user_input = ""

# Initialize Vector Store
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


# Solara Components

@solara.component
def UploadSection():
    global uploaded_files
    uploaded_files = solara.file_uploader("Upload Policy Documents (PDF)", multiple=True, accept=".pdf")
    if uploaded_files:
        solara.notification("Processing uploaded documents...")
        global vectordb
        vectordb = initialize_vector_store(uploaded_files)
        solara.notification("Documents processed successfully!", type="success")


@solara.component
def ChatBot():
    global user_input
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
        solara.Markdown("### Chat with the Insurance Bot")
        user_input = solara.Text("Ask a question about the uploaded insurance policies:")
        if user_input:
            result = qa_chain({"query": user_input})
            solara.Markdown(f"### Answer:\n{result['result']}")

            # Display Source Document (Optional)
            with solara.Expand("Source Document"):
                source_doc = result["source_documents"][0]
                solara.Markdown(source_doc.page_content)
    else:
        solara.Info("Please upload documents to enable the chatbot.")


@solara.component
def App():
    solara.Markdown("# Insurance Policy Query Chatbot")
    UploadSection()
    ChatBot()


# Run Solara App
if __name__ == "__main__":
    solara.run(App)

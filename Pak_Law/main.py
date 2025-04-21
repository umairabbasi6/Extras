import os
import streamlit as st
import faiss
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# Ignore warnings
warnings.filterwarnings("ignore")
load_dotenv()

# Initialize Streamlit
st.set_page_config(page_title="RAG-based Q&A", layout="wide")
st.title("📄 RAG-based Q&A System with Ollama and FAISS")

st.write("Loading pre-trained PDF documents...")

# Load pre-existing PDFs from the 'Data' directory
pdf_dir = "Data"
pdfs = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir) if file.endswith('.pdf')]

if not pdfs:
    st.error("No PDF documents found in the 'Data' directory. Please add some and restart the app.")
    st.stop()

st.write(f"Found {len(pdfs)} PDFs. Processing...")

# Load PDFs
docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    docs.extend(pages)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# Embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

# Create FAISS index
single_vector = embeddings.embed_query("sample text")
index = faiss.IndexFlatL2(len(single_vector))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add documents to vector store
vector_store.add_documents(documents=chunks)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 1})

st.success("PDFs processed successfully! You can now ask questions.")

# Set up the LLM
model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")

# Prompt template
prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    - If the answer is not within the context, clearly state that you don't know.
    - Refuse to answer questions that are irrelevant to the provided context.
    - Answer in bullet points for clarity.
    - Make sure your answer is relevant to the question and derived solely from the given data that I provided.
    Question: {question}
    Context: {context}
    Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# User input for Q&A
question = st.text_input("Ask a question about the preloaded documents:")

if question:
    with st.spinner("Retrieving answer..."):
        output = rag_chain.invoke(question)
        st.subheader("Answer:")
        st.write(output)

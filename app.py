import streamlit as st
import os
import shutil

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List

TEMP_DIR = os.path.join(os.getcwd(), "temp")

os.makedirs(TEMP_DIR, exist_ok=True)

st.title("PDF Chat")

st.sidebar.header("Configuration")

GROQ_API_KEY = st.sidebar.text_input("Groq API KEY", type="password")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore(embedding=embeddings)
uploaded_file = st.file_uploader(label="Upload PDF File", type="pdf", accept_multiple_files=False)

llm = None

def create_embeddings(pdf_file):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    splits = splitter.split_documents(docs)
    _ = vectorstore.add_documents(splits)
     

if uploaded_file:
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    create_embeddings(file_path)

if GROQ_API_KEY:
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        api_key=GROQ_API_KEY
    )

question = st.text_input("Enter your question")

class State(TypedDict):
  question: str
  context: List[Document]
  answer: str

prompt = hub.pull("rlm/rag-prompt")

def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def create_graph_app():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph

if question:
    if not GROQ_API_KEY:
        st.error("No groq api provided")
    if not uploaded_file:
        st.error("Please provide a pdf file first")
    
    app = create_graph_app()
    response = app.invoke({"question": question})
    st.markdown(response["answer"])

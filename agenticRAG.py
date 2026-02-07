import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

# ----------- UPDATED IMPORTS -----------
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

from langgraph.prebuilt import create_react_agent as create_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import List

from docx import Document

# ----------------- ENV -----------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")   # only for embeddings

if not PINECONE_API_KEY:
    st.error("Missing Pinecone API key.")
    st.stop()

# ----------------- PINECONE -----------------

def delete_vectors():
    index = Pinecone(api_key=PINECONE_API_KEY).Index("myindex")
    return index.delete(delete_all=True)

if "cleaned" not in st.session_state:
    try:
        delete_vectors()
        st.session_state.cleaned = True
    except:
        st.session_state.cleaned = True


# ----------------- FILE PROCESSING -----------------

def process_files(files):
    documents = []

    for file in files:
        filename = file.name.lower()

        if filename.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
            os.remove(tmp_path)

        elif filename.endswith(('.docx', '.doc')):
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])

            DocumentObj = type("Document", (object,), {})
            doc_obj = DocumentObj()

            setattr(doc_obj, "page_content", text)
            setattr(doc_obj, "metadata", {"source": file.name})

            documents.append(doc_obj)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_documents(documents)


def add_docs_vectordb(files: List, vector_store):
    all_splits = process_files(files)
    return vector_store.add_documents(documents=all_splits)


# ----------------- RETRIEVAL TOOL -----------------

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve relevant document chunks from Pinecone.
    """

    docs = vector_store.similarity_search(query, k=3)

    text = "\n\n".join(
        f"Source: {d.metadata}\nContent: {d.page_content}"
        for d in docs
    )

    return text, docs


# ----------------- LLM -----------------

def get_ai_response(input_message, graph, config):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": input_message}]},
        config=config
    )
    return result["messages"][-1].content


# ----------------- INIT -----------------

# LOCAL MISTRAL MODEL
llm = ChatOllama(
    model="mistral",
    temperature=0.1
)

# Keep OpenAI only for embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
    dimensions=3072
)

index = Pinecone(api_key=PINECONE_API_KEY).Index("myindex")

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index
)

memory = MemorySaver()

agent_executor = create_agent(llm, [retrieve], checkpointer=memory)

config = {"configurable": {"thread_id": "local-demo"}}


# ----------------- UI -----------------

st.title("Custom RAG – Mistral Local")

uploaded_files = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "doc"],
    accept_multiple_files=True
)

if st.button("Add Documents to VectorDB"):
    add_docs_vectordb(uploaded_files, vector_store)
    st.success("Added!")


prompt = st.chat_input("Ask...")

if prompt:
    with st.spinner("Mistral thinking..."):
        response = get_ai_response(prompt, agent_executor, config)

    st.write(response)
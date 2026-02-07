import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

# ----------- UPDATED IMPORTS -----------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

from langgraph.prebuilt import create_react_agent as create_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import List

from docx import Document

# ----------------- ENV -----------------

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("App not configured correctly. Missing API keys.")
    st.stop()

# ----------------- PINECONE -----------------

def delete_vectors():
    index = Pinecone(api_key=PINECONE_API_KEY).Index("myindex")
    return index.delete(delete_all=True)


# AUTO DELETE ONCE PER SESSION
if "cleaned" not in st.session_state:
    try:
        delete_vectors()
        st.session_state.cleaned = True
        st.info("Fresh session started – previous vectors cleared.")
    except:
        st.session_state.cleaned = True


def create_pinecone_index(index_name, dimension):
    metric = "cosine"
    pc = Pinecone(api_key=PINECONE_API_KEY)

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    return pc.Index(index_name)


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

        else:
            st.warning(f"Unsupported file type: {file.name}. Only PDF and Word files are supported.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    return text_splitter.split_documents(documents)


def add_docs_vectordb(files: List, vector_store):
    all_splits = process_files(files)
    return vector_store.add_documents(documents=all_splits)


# ----------------- RETRIEVAL TOOL -----------------

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve relevant document chunks from Pinecone based on user query.
    Used by the LangGraph agent to perform RAG.
    """

    retrieved_docs = vector_store.similarity_search(query, k=3)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


# ----------------- LLM -----------------

def get_ai_response(input_message, graph, config):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": input_message}]},
        config=config
    )
    return result["messages"][-1].content


# ----------------- INIT -----------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

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

config = {"configurable": {"thread_id": "public-demo"}}


# ----------------- UI -----------------

st.set_page_config(page_title="Custom RAG", page_icon="📝")

with st.sidebar:
    st.header("Upload")

    uploaded_files = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True
    )

    if st.button("Add Documents to VectorDB"):
        if uploaded_files:
            add_docs_vectordb(uploaded_files, vector_store)
            st.success("Documents added to the vector store!")
        else:
            st.warning("Please upload a document first.")


st.title("Custom RAG! Chat with Notes!")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hey There! I'm here to assist you answer your questions."
    }]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt = st.chat_input("Ask me anything about your document...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        response = get_ai_response(prompt, agent_executor, config)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

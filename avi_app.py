import streamlit as st
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
DOCS_FOLDER = "Avi_Data"  # Folder containing your PDFs
DB_PATH = "avi_db"        # Folder for the vector database storage
MODEL_NAME = "llama3.1:8b" # Change to "llama3.2:1b-instruct-q4_K_M" for speed
EMBEDDING_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Avi - Personal Assistant", layout="wide")

# --- backend Functions ---

@st.cache_resource
def load_and_process_pdfs():
    """Loads PDFs, splits them, and creates/updates the Vector DB."""
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        return None

    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_files:
        return None

    documents = []
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    # Split text into chunks (Avi reads in small bites)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Embed and store in ChromaDB (Persistent)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    return vector_db

def get_response_from_avi(query, vector_db):
    """Retrieves context and generates a response."""
    llm = ChatOllama(model=MODEL_NAME)
    
    # Retrieve relevant chunks from the database
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    context_docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    # Prompt Engineering
    template = """
    You are Avi, a helpful and secure personal assistant. 
    Answer the question based ONLY on the following context from the user's personal documents:
    
    {context}
    
    Question: {question}
    
    If the answer is not in the context, politely say you don't have that information in the documents.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    return chain.invoke({"context": context_text, "question": query})

# --- Frontend (UI) ---

st.title("🤖 Avi: Personal Data Assistant")
st.markdown(f"**Status:** Running locally on *{MODEL_NAME}*")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for System Ops
with st.sidebar:
    st.header("🧠 Knowledge Base")
    if st.button("Refresh Documents"):
        with st.spinner("Avi is reading your PDFs..."):
            db = load_and_process_pdfs()
            if db:
                st.session_state.vector_db = db
                st.success("Documents processed!")
            else:
                st.warning("No PDFs found in 'Avi_Data' folder.")
    
    st.info("Place your PDF files in the 'Avi_Data' folder and click Refresh.")

# Load DB if not loaded
if "vector_db" not in st.session_state:
    if os.path.exists(DB_PATH):
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        st.session_state.vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        st.warning("Please add documents and click 'Refresh Documents' to start.")

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask Avi about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if "vector_db" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Avi is thinking..."):
                response_msg = get_response_from_avi(prompt, st.session_state.vector_db)
                st.markdown(response_msg.content)
                st.session_state.messages.append({"role": "assistant", "content": response_msg.content})
    else:
        st.error("Avi needs knowledge! Please refresh documents first.")
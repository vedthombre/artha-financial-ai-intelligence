import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "chroma_db"

def ingest_data(file_path, file_name):
    """Loads PDF, splits into chunks, and saves to Vector DB."""
    # 1. Load PDF
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Loading {file_name}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs)

    # 3. Add Metadata
    for doc in doc_splits:
        doc.metadata["source"] = file_name

    # 4. Embed & Store
    print(f"Embedding {len(doc_splits)} chunks...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
        persist_directory=DB_DIR,
        collection_name="rag-chroma"
    )
    print("Ingestion Complete.")
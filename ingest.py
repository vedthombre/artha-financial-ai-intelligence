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
    if not docs:
        raise ValueError(f"No readable pages found in {file_name}")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs)
    if not doc_splits:
        raise ValueError(f"No text content found in {file_name}")

    # 3. Add Metadata
    for doc in doc_splits:
        doc.metadata["source"] = file_name

    # 4. Embed & Store
    print(f"Embedding {len(doc_splits)} chunks...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_name="rag-chroma",
    )
    existing = store.get(
        where={"source": file_name},
        include=["documents"],
    )
    existing_documents = existing.get("documents") or []
    existing_ids = existing.get("ids") or []
    new_contents = [doc.page_content for doc in doc_splits]
    if len(existing_ids) == len(new_contents) and set(existing_documents) == set(new_contents):
        print("Ingestion skipped: document is already indexed.")
        return

    Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
        persist_directory=DB_DIR,
        collection_name="rag-chroma"
    )
    if existing_ids:
        store.delete(ids=existing_ids)
    print("Ingestion Complete.")
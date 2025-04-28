# ingest.py

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from db.vector_store import get_storage_context
from models.embeddings import get_embed_model


def ingest_documents(docs_dir="./data/docs"):
    """Ingest documents from the specified directory into the vector store."""
    print(f"Loading documents from {docs_dir}...")

    # Check if directory exists and has files
    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} does not exist. Creating it...")
        os.makedirs(docs_dir, exist_ok=True)
        print(f"Please add your documents to {docs_dir} and run this script again.")
        return

    if not os.listdir(docs_dir):
        print(f"No files found in {docs_dir}. Please add documents and run again.")
        return

    # Get the storage context and embedding model
    storage_context, _ = get_storage_context()
    embed_model = get_embed_model()

    # Load documents and create the index
    documents = SimpleDirectoryReader(docs_dir).load_data()
    print(f"Loaded {len(documents)} documents. Processing...")

    # Create the index with the documents
    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )

    print("Documents successfully ingested into the vector store.")
    return vector_index


if __name__ == "__main__":
    ingest_documents()

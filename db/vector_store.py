# db/vector_store.py

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CHROMA_DB_PATH, COLLECTION_NAME


def get_vector_store():
    """Initialize and return the ChromaDB vector store."""
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store


def get_storage_context():
    """Get the storage context with the vector store."""
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context, vector_store

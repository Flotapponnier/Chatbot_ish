# models/embeddings.py

from llama_index.embeddings.ollama import OllamaEmbedding
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OLLAMA_BASE_URL, EMBEDDING_MODEL


def get_embed_model():
    """Initialize and return the embedding model."""
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return embed_model

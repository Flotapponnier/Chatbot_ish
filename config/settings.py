# config/settings.py

# Database settings
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "ishi_collection"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "ishi"
LLM_TIMEOUT = 60.0

# Retriever settings
SIMILARITY_TOP_K = 5

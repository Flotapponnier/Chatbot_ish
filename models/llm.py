# models/llm.py

from llama_index.llms.ollama import Ollama
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OLLAMA_BASE_URL, LLM_MODEL, LLM_TIMEOUT


def get_llm():
    """Initialize and return the LLM."""
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=LLM_TIMEOUT,
    )
    return llm

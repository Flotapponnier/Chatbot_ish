# chat.py

import time
import sys
import threading
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine

from db.vector_store import get_storage_context
from models.embeddings import get_embed_model
from models.llm import get_llm
from config.settings import SIMILARITY_TOP_K


def initialize_retriever_and_engine():
    """Initialize and return the retriever and query engine."""
    # Get components
    storage_context, vector_store = get_storage_context()
    embed_model = get_embed_model()
    llm = get_llm()

    # Load the vector index
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )

    # Create a retriever
    retriever = vector_index.as_retriever(
        similarity_top_k=SIMILARITY_TOP_K,
    )

    # Create a query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )

    return retriever, query_engine, llm


def stream_response(response_text):
    """Display response text progressively, word by word."""
    print("\nIshi:", end=" ", flush=True)

    # Split the text into words
    words = response_text.split()

    # Display words progressively
    for i, word in enumerate(words):
        print(word, end=" ", flush=True)

        # Determine delay based on punctuation
        if any(p in word for p in [".", "!", "?"]):
            time.sleep(0.4)  # Longer pause after sentences
        elif any(p in word for p in [",", ";", ":"]):
            time.sleep(0.25)  # Medium pause after clauses
        else:
            time.sleep(0.1)  # Standard delay between words

    print()  # End with a newline


def process_query(query, retriever, query_engine, llm):
    """Process query, first with documents then fallback to LLM."""
    try:
        # Check if we have relevant documents
        nodes = retriever.retrieve(query)

        if nodes:
            # We found documents, use them
            response = query_engine.query(query)
            return str(response), True
        else:
            # No relevant documents found, use direct LLM
            response = llm.complete(query)
            return str(response), False

    except Exception as e:
        print(f"\nError during processing: {e}")
        # Fallback to direct LLM
        try:
            response = llm.complete(query)
            return str(response), False
        except Exception as fallback_e:
            return f"I'm sorry, I encountered an error: {fallback_e}", False


def chat_loop():
    """Run the interactive chat loop with progressive text display."""
    print("Welcome to Ishi Chatbot! Type 'exit' to quit.")

    # Initialize components
    retriever, query_engine, llm = initialize_retriever_and_engine()

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("\nIshi: Goodbye! Have a great day!")
            break

        # Show thinking indicator
        print("\nIshi is thinking...", end="", flush=True)

        # Process the query
        response_text, used_docs = process_query(
            user_input, retriever, query_engine, llm
        )

        # Add a prefix if using general knowledge
        if not used_docs:
            response_text = "Based on my general knowledge: " + response_text

        print("\r                       \r", end="", flush=True)

        # Stream the response
        stream_response(response_text)


if __name__ == "__main__":
    chat_loop()

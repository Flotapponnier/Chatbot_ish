# chat.py

import time
import sys
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


def print_step_by_step(text, delay=0.03):
    """Print text gradually to simulate step-by-step typing."""
    print("\nIshi: ", end="", flush=True)

    # Split the text into sentences for more natural pauses
    sentences = text.split(". ")
    for i, sentence in enumerate(sentences):
        # Add period back except for the last sentence if it doesn't end with punctuation
        if i < len(sentences) - 1:
            sentence += "."

        # Print each character with a slight delay
        for char in sentence:
            print(char, end="", flush=True)
            time.sleep(delay)

        # Add a slightly longer pause between sentences
        if i < len(sentences) - 1:
            time.sleep(0.2)

    print()  # Add a newline at the end


def process_with_document(query, query_engine):
    """Process a query using document knowledge and return response."""
    print_step_by_step("Searching my knowledge base...", 0.02)
    response = query_engine.query(query)
    print_step_by_step("Here's what I found:", 0.02)
    return str(response)


def process_with_llm(query, llm):
    """Process a query using only the LLM and return response."""
    print_step_by_step("Thinking about your question...", 0.02)
    response = llm.complete(query)
    return str(response)


def chat_loop():
    """Run the interactive chat loop."""
    print("Welcome to Ishi Chatbot! Type 'exit' to quit.")

    # Initialize components
    retriever, query_engine, llm = initialize_retriever_and_engine()

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print_step_by_step("Goodbye! Have a great day!")
            break

        try:
            # Check if there are relevant documents
            nodes = retriever.retrieve(user_input)

            if nodes:
                # Documents found, use them to answer
                response = process_with_document(user_input, query_engine)
                print_step_by_step(response)
            else:
                # No relevant documents, use direct LLM
                print_step_by_step(
                    "I don't have specific information about that in my knowledge base."
                )
                response = process_with_llm(user_input, llm)
                print_step_by_step(response)

        except Exception as e:
            print(f"\nError: {e}")
            print_step_by_step(
                "I encountered a problem processing your request. Let me try a different approach."
            )

            try:
                # Fallback to direct LLM
                response = process_with_llm(user_input, llm)
                print_step_by_step(response)
            except Exception as fallback_error:
                print(f"\nFallback error: {fallback_error}")
                print_step_by_step(
                    "I'm sorry, but I'm having technical difficulties at the moment. Please try again."
                )


if __name__ == "__main__":
    chat_loop()

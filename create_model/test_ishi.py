import ollama
import os
import signal
import sys
from datetime import datetime
import model  # Import your existing model module

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nThank you for using Ishi. Take care! 🌱")
    sys.exit(0)

def chat_with_ishi():
    """Main function to chat with Ishi."""
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get model name and options from your module
    model_name = model.model_name
    options = model.options
    
    # Chat history
    messages = []
    
    # Welcome message
    clear_screen()
    print("=" * 65)
    print("    Welcome to Ishi - 42 School Mental Health Resource Assistant")
    print("=" * 65)
    print("\nIshi is here to provide mental health resources and support.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.\n")
    print("=" * 65)
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("\n📝 You: ")
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using Ishi. Take care! 🌱")
            break
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Show "thinking" indicator
        print("\n🤔 Ishi is thinking...", end="", flush=True)
        
        try:
            # Get response from model
            start_time = datetime.now()
            response = ollama.chat(model=model_name, messages=messages, options=options)
            end_time = datetime.now()
            
            # Calculate response time
            response_time = (end_time - start_time).total_seconds()
            
            # Clear the "thinking" indicator
            print("\r" + " " * 20 + "\r", end="", flush=True)
            
            # Extract and display the response
            assistant_message = response['message']['content']
            print(f"\n🤖 Ishi: {assistant_message}")
            print(f"\n(Response time: {response_time:.2f}s)")
            print("-" * 65)
            
            # Add assistant message to history
            messages.append({"role": "assistant", "content": assistant_message})
            
        except Exception as e:
            print(f"\r\n❌ Error: {e}")
            print("Let's try again. If the problem persists, please restart the program.")

if __name__ == "__main__":
    chat_with_ishi()

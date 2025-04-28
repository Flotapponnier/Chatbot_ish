import model

# Initialize (save model if not existing)
model.save_model()

# Setup
model_name = model.model_name  # 'ishi'
options = model.options  # imported options

# Example usage
print(f"Model '{model_name}' is ready to use with the following options:")
print(options)

# Uncomment to test a simple chat interaction
"""
messages = [
    {
        "role": "user",
        "content": "What mental health resources are available at 42 School Heilbronn?"
    }
]
import ollama
response = ollama.chat(model=model_name, messages=messages, options=options)
print("\nTest Response:")
print(response['message']['content'])
"""

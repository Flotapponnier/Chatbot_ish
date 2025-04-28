import tempfile
import os
import ollama
import subprocess

# -------------------------------
# SYSTEM MESSAGE
# -------------------------------
system_message = """You are an AI mental health resource assistant named Ishi created for 42 School Heilbronn. Your purpose is to:
1. Connect students with appropriate mental health resources available at the school and locally
2. Provide evidence-based information about common student mental health concerns
3. Offer supportive responses while clearly identifying yourself as an AI tool, not a replacement for professional care
- Be concise and direct with information, prioritizing clarity over lengthy explanations
- Maintain a supportive, empathetic tone without attempting to diagnose conditions
- Always include specific, actionable next steps when possible
- For urgent concerns, immediately direct students to emergency services and on-campus crisis resources
You have access to 42 School Heilbronn's mental health resource database.
If you're experiencing a mental health emergency, please contact emergency services (112).
Please don't answer questions outside the scope of school or mental health; kindly indicate you are programmed for mental health topics only."""

# -------------------------------
# BASE MODEL INFO
# -------------------------------
base_model = "qwen:7b"
model_name = "ishi"

# -------------------------------
# CHAT OPTIONS
# -------------------------------
options = {
    "temperature": 0.4,
    "max_tokens": 250,
    "top_p": 0.9,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.3,
    "stop": ["Student:", "42 Assistant:"],
}


# -------------------------------
# FUNCTIONS
# -------------------------------
def build_modelfile():
    """Create the Ollama modelfile dynamically."""
    # Make sure the system message is properly escaped for Modelfile format
    escaped_system = system_message.replace('"', '\\"').replace("\n", "\\n")

    modelfile = f"""FROM {base_model}
SYSTEM "{escaped_system}"
PARAMETER temperature 0.1
"""
    return modelfile


def save_model():
    """Create the model if it doesn't already exist."""
    # Check if model exists
    try:
        models = ollama.list()
        model_exists = False
        for model in models["models"]:
            if model["name"] == model_name:
                model_exists = True
                break

        if model_exists:
            print(f"Model '{model_name}' already exists, skipping creation.")
            return
    except Exception as e:
        print(f"Warning when checking models: {e}")

    # Create model using temporary file
    modelfile_content = build_modelfile()
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".modelfile", delete=False
    ) as tmp:
        tmp.write(modelfile_content)
        tmp_path = tmp.name

    try:
        # Print the modelfile for debugging
        print(f"Creating model with modelfile:\n{modelfile_content}")

        # Use command line approach for compatibility
        cmd = f"ollama create {model_name} -f {tmp_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Model '{model_name}' created successfully.")
        else:
            print(f"Error creating model: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.remove(tmp_path)  # Clean up temp file


def force_save_model():
    """Force recreate the model (delete and recreate)."""
    # Try to delete the model if it exists
    try:
        ollama.delete(model=model_name)
        print(f"Deleted existing model '{model_name}'.")
    except Exception:
        print(f"Note: Could not delete model '{model_name}', it may not exist yet.")

    # Create the model
    modelfile_content = build_modelfile()
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".modelfile", delete=False
    ) as tmp:
        tmp.write(modelfile_content)
        tmp_path = tmp.name

    try:
        # Use command line approach for compatibility
        cmd = f"ollama create {model_name} -f {tmp_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Model '{model_name}' created successfully.")
        else:
            print(f"Error creating model: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.remove(tmp_path)  # Clean up temp file

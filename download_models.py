# =============================================================================
# download_models.py (Memory-Safe Version)
# =============================================================================
# Goal: A one-time script that ONLY downloads model files to the cache
# without loading them into memory, preventing crashes on systems with limited RAM.

from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download # Import the correct tool for the job
import os

# --- Configuration: List all the models your app will ever need ---
EMBEDDING_MODELS = [
    "HooshvareLab/bert-base-parsbert-uncased",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]
LLM_MODEL = 'mshojaei77/gemma-2-2b-fa-v2' # The name you had before was slightly different, let's stick to the one in the app.py

# To use a secret token from your environment (more secure)
# You might need to log in via `huggingface-cli login` first in your terminal
# from dotenv import load_dotenv
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

def download_all_models():
    """Iterates through the models and downloads them to the local cache."""
    print("--- Starting Model Download Process ---")
    
    # Download Embedding Models
    for model_name in EMBEDDING_MODELS:
        try:
            print(f"\nDownloading embedding model: {model_name}...")
            # This line is fine, SentenceTransformer is usually memory-efficient on init
            SentenceTransformer(model_name)
            print(f"✅ Successfully downloaded and cached {model_name}.")
        except Exception as e:
            print(f"❌ Failed to download {model_name}. Error: {e}")

    # Download the Intermediary LLM (The Memory-Safe Way)
    try:
        print(f"\nDownloading interpreter LLM: {LLM_MODEL}...")
        print("(This is a large model and may take a long time to download.)")
        
        ### --- THE KEY CHANGE IS HERE --- ###
        # snapshot_download ONLY downloads the files from the repository.
        # It does NOT try to load the model into your computer's RAM.
        snapshot_download(
            repo_id=LLM_MODEL,
            # token=HF_TOKEN, # Uncomment if you have issues with private models or auth
            ignore_patterns=["*.safetensors.index.json", "*.onnx", "*.h5"] # Ignore files we don't need
        )
        ### --- END OF KEY CHANGE --- ###
        
        print(f"✅ Successfully downloaded and cached {LLM_MODEL}.")
    except Exception as e:
        print(f"❌ Failed to download {LLM_MODEL}. Error: {e}")
        
    print("\n--- Model download process complete! ---")
    print("You can now run 'app.py'. It will load the models from your local cache.")

if __name__ == "__main__":
    download_all_models()
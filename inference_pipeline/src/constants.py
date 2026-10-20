import os

# Qdrant
QDRANT_URL = "https://YOUR-CLUSTER-URL:6333"  # change this
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "finance_news"
QDRANT_TEXT_KEY = "text"  # payload field holding the article text

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_LLM = "mistralai/Mistral-7B-Instruct-v0.2"
FALLBACK_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # used when no GPU

# Comet model registry (LoRA adapter)
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
COMET_PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")
COMET_MODEL_NAME = "finance-mistral-lora"  # registry name, change to yours
COMET_MODEL_VERSION = "latest"
ADAPTER_DIR = "./adapter"

# Set USE_ADAPTER=false to run the base model before training is done
USE_ADAPTER = os.getenv("USE_ADAPTER", "true").lower() == "true"

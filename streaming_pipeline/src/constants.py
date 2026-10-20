import os

# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

# Qdrant (must match the inference pipeline)
QDRANT_URL = "https://YOUR-CLUSTER-URL:6333"  # change this
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "finance_news"
QDRANT_TEXT_KEY = "text"

# Embeddings (must match the inference pipeline)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunking, in characters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

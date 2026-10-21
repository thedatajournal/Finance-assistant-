# Streaming Pipeline

Collects financial news from Alpaca, converts it into embeddings, and stores it in Qdrant. The inference pipeline later searches this data to answer questions.

## How It Works

1. **Fetch:** `alpaca_batch.py` pulls historical news, and `alpaca_stream.py` listens to the live news websocket.
2. **Chunk:** each article is cleaned of HTML and split into overlapping chunks.
3. **Embed:** chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` (normalized).
4. **Store:** vectors and metadata are upserted into the `finance_news` Qdrant collection. IDs are deterministic, so re-running never creates duplicates.

## Running Locally

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
QDRANT_API_KEY=your_key
```

Set `QDRANT_URL` in `src/constants.py`, then:

```bash
bash run_batch.sh 7          # backfill the last 7 days
bash run_realtime_dev.sh     # stream live news
```

## Docker

```bash
docker compose up --build
```

## Notes

The embedding model, collection name, and `text` payload field must match the inference pipeline.

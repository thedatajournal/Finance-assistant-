import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sentence_transformers import SentenceTransformer

from src import constants
from src.alpaca_batch import fetch_news
from src.alpaca_stream import run_stream
from src.models import NewsArticle
from src.qdrant import ensure_collection, get_client, upsert_chunks
from src.utils import chunk_text, make_point_id, setup_logging

logger = logging.getLogger(__name__)


class Ingestor:
    """Chunks articles, embeds them, and stores them in Qdrant."""

    def __init__(self):
        self.model = SentenceTransformer(constants.EMBEDDING_MODEL)
        self.client = get_client()
        ensure_collection(self.client)

    def ingest(self, articles: list) -> int:
        ids, texts, payloads = [], [], []
        for article in articles:
            chunks = chunk_text(
                article.to_text(), constants.CHUNK_SIZE, constants.CHUNK_OVERLAP
            )
            for i, chunk in enumerate(chunks):
                ids.append(make_point_id(article.id, i))
                texts.append(chunk)
                payloads.append(article.payload(chunk, i))

        if not texts:
            return 0

        vectors = self.model.encode(
            texts, normalize_embeddings=True, batch_size=32
        ).tolist()
        upsert_chunks(self.client, ids, vectors, payloads)
        return len(texts)


def run_batch(ingestor: Ingestor, days: int) -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    articles = fetch_news(start, end)
    logger.info("Fetched %d articles", len(articles))

    total = 0
    for i in range(0, len(articles), 50):
        total += ingestor.ingest(articles[i:i + 50])
    logger.info("Stored %d chunks", total)


def run_realtime(ingestor: Ingestor) -> None:
    async def handler(news):
        article = NewsArticle.from_alpaca(news)
        stored = await asyncio.to_thread(ingestor.ingest, [article])
        logger.info("Stored %d chunks: %s", stored, article.headline)

    logger.info("Listening for live news...")
    run_stream(handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "realtime"], required=True)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    setup_logging()
    ingestor = Ingestor()

    if args.mode == "batch":
        run_batch(ingestor, args.days)
    else:
        run_realtime(ingestor)


if __name__ == "__main__":
    main()

import logging
import re
import uuid
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "streaming.log"),
        ],
    )


def clean_text(text) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int, overlap: int) -> list:
    """Split text into overlapping character chunks."""
    if not text:
        return []
    chunks, start = [], 0
    while True:
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def make_point_id(article_id: str, chunk_index: int) -> str:
    """Deterministic ID so re-running never creates duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{article_id}-{chunk_index}"))

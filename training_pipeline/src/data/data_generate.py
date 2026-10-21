import argparse
import logging
import os
import random

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.data.dataset import format_context
from src.utils import DATA_DIR, load_config, read_json, setup_logging, write_json

logger = logging.getLogger(__name__)

TEACHER_SYSTEM = (
    "You are an expert financial analyst. Answer the question using only the "
    "news articles provided. Be concise (2-4 sentences). If the articles do not "
    "contain enough information, say so instead of guessing."
)


def retrieve_context(client, embedder, cfg: dict, question: str) -> list:
    """Fetch the most relevant news chunks from Qdrant."""
    text_key = cfg["qdrant"]["text_key"]
    vector = embedder.encode(question, normalize_embeddings=True).tolist()
    points = client.query_points(
        collection_name=cfg["qdrant"]["collection"],
        query=vector,
        limit=cfg["data"]["top_k"],
        with_payload=True,
    ).points
    return [p.payload[text_key] for p in points if text_key in p.payload]


def ask_teacher(llm, cfg: dict, question: str, context: str) -> str:
    response = llm.chat.completions.create(
        model=cfg["teacher"]["model"],
        temperature=cfg["teacher"]["temperature"],
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM},
            {
                "role": "user",
                "content": f"News articles:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(DATA_DIR / "input_queries.json"))
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()

    client = QdrantClient(
        url=cfg["qdrant"]["url"], api_key=os.getenv("QDRANT_API_KEY")
    )
    embedder = SentenceTransformer(cfg["data"]["embedding_model"])
    llm = OpenAI()  # reads OPENAI_API_KEY from the environment

    records = []
    for question in read_json(args.queries):
        docs = retrieve_context(client, embedder, cfg, question)
        if not docs:
            logger.warning("No context found, skipping: %s", question)
            continue
        context = format_context(docs)
        try:
            answer = ask_teacher(llm, cfg, question, context)
        except Exception as e:
            logger.error("Teacher failed for %r: %s", question, e)
            continue
        records.append({"question": question, "context": context, "answer": answer})

    if not records:
        logger.error("No records generated. Is Qdrant filled? Run the streaming pipeline first.")
        return

    random.Random(cfg["data"]["seed"]).shuffle(records)
    n_test = max(1, int(len(records) * cfg["data"]["test_size"]))
    test, train = records[:n_test], records[n_test:]

    version = cfg["data"]["version"]
    write_json(DATA_DIR / f"training_data_{version}.json", train)
    write_json(DATA_DIR / f"test_data_{version}.json", test)
    logger.info("Saved %d training and %d test examples", len(train), len(test))


if __name__ == "__main__":
    main()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src import constants


def get_client() -> QdrantClient:
    return QdrantClient(url=constants.QDRANT_URL, api_key=constants.QDRANT_API_KEY)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if constants.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=constants.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=constants.EMBEDDING_DIM, distance=Distance.COSINE
            ),
        )


def upsert_chunks(client: QdrantClient, ids: list, vectors: list, payloads: list) -> None:
    points = [
        PointStruct(id=i, vector=v, payload=p)
        for i, v, p in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=constants.QDRANT_COLLECTION, points=points)

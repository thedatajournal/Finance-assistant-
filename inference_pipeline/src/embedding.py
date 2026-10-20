from langchain_community.embeddings import HuggingFaceEmbeddings

from src import constants
from src.utils import get_device


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=constants.EMBEDDING_MODEL,
        model_kwargs={"device": get_device()},
        encode_kwargs={"normalize_embeddings": True},
    )

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src import constants
from src.utils import load_config

PROMPT_TEMPLATE = """You are a finance assistant. Answer the question using the news articles below. If the articles don't contain the answer, say so instead of guessing.

News articles:
{context}

Conversation so far:
{history}

Question: {question}"""


def build_retriever(embeddings):
    client = QdrantClient(url=constants.QDRANT_URL, api_key=constants.QDRANT_API_KEY)
    store = QdrantVectorStore(
        client=client,
        collection_name=constants.QDRANT_COLLECTION,
        embedding=embeddings,
        content_payload_key=constants.QDRANT_TEXT_KEY,
    )
    top_k = load_config()["retrieval"]["top_k"]
    return store.as_retriever(search_kwargs={"k": top_k})


def format_docs(docs) -> str:
    if not docs:
        return "No relevant news found."
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def build_chain(retriever, llm, tokenizer):
    def to_prompt(inputs: dict) -> str:
        content = PROMPT_TEMPLATE.format(**inputs)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | RunnableLambda(to_prompt)
        | llm
        | StrOutputParser()
    )

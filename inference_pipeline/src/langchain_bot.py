from src.chains import build_chain, build_retriever
from src.embedding import build_embeddings
from src.model import build_llm
from src.utils import format_history, load_config


class FinanceBot:
    def __init__(self):
        self.turns = load_config()["chat"]["history_turns"]
        embeddings = build_embeddings()
        llm, tokenizer = build_llm()
        retriever = build_retriever(embeddings)
        self.chain = build_chain(retriever, llm, tokenizer)

    def ask(self, question: str, history: list) -> str:
        answer = self.chain.invoke(
            {
                "question": question,
                "history": format_history(history, self.turns),
            }
        )
        return answer.strip()

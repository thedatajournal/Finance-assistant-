from src.langchain_bot import FinanceBot

_bot = None


def get_bot() -> FinanceBot:
    global _bot
    if _bot is None:
        _bot = FinanceBot()
    return _bot


def respond(message: str, history: list) -> str:
    try:
        return get_bot().ask(message, history)
    except Exception as e:
        return f"Something went wrong: {e}"

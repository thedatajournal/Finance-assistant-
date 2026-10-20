from alpaca.data.live.news import NewsDataStream

from src import constants


def run_stream(handler) -> None:
    """Listen to the live Alpaca news websocket. Blocks until stopped."""
    stream = NewsDataStream(
        constants.ALPACA_API_KEY,
        constants.ALPACA_API_SECRET,
    )
    stream.subscribe_news(handler, "*")
    stream.run()

from datetime import datetime
from typing import Optional

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from src import constants
from src.models import NewsArticle


def fetch_news(
    start: datetime,
    end: datetime,
    symbols: Optional[str] = None,
    limit: int = 500,
) -> list:
    """Fetch historical news articles from Alpaca."""
    client = NewsClient(
        api_key=constants.ALPACA_API_KEY,
        secret_key=constants.ALPACA_API_SECRET,
    )
    request = NewsRequest(
        start=start,
        end=end,
        symbols=symbols,
        limit=limit,
        include_content=True,
    )
    news_set = client.get_news(request)
    return [NewsArticle.from_alpaca(n) for n in news_set.data.get("news", [])]

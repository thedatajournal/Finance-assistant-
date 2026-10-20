from dataclasses import dataclass, field

from src import constants
from src.utils import clean_text


@dataclass
class NewsArticle:
    id: str
    headline: str
    summary: str
    content: str
    author: str
    source: str
    url: str
    symbols: list = field(default_factory=list)
    created_at: str = ""

    @classmethod
    def from_alpaca(cls, news) -> "NewsArticle":
        return cls(
            id=str(news.id),
            headline=clean_text(news.headline),
            summary=clean_text(news.summary),
            content=clean_text(news.content),
            author=news.author or "",
            source=news.source or "",
            url=news.url or "",
            symbols=list(news.symbols or []),
            created_at=str(news.created_at),
        )

    def to_text(self) -> str:
        parts = [self.headline, self.content or self.summary]
        return ". ".join(p for p in parts if p)

    def payload(self, chunk: str, chunk_index: int) -> dict:
        return {
            constants.QDRANT_TEXT_KEY: chunk,
            "article_id": self.id,
            "chunk_index": chunk_index,
            "headline": self.headline,
            "source": self.source,
            "url": self.url,
            "symbols": self.symbols,
            "created_at": self.created_at,
        }

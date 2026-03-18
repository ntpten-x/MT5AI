from __future__ import annotations

import asyncio
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import quote_plus

import httpx
from loguru import logger

DEFAULT_GOOGLE_NEWS_QUERY = (
    '"Federal Reserve" OR inflation OR "interest rates" OR recession OR '
    '"Treasury yields" OR "gold prices" OR ETF OR "S&P 500"'
)


@dataclass(slots=True, frozen=True)
class NewsArticle:
    title: str
    link: str
    source: str | None
    published_at: datetime | None
    summary: str | None
    guid: str | None


class NewsClient:
    """Fetches macro and market news from free RSS feeds."""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        user_agent: str = "invest-advisor-bot/0.1",
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent

    async def fetch_latest_macro_news(
        self,
        *,
        limit: int = 10,
        when: str = "1d",
    ) -> list[NewsArticle]:
        feed_url = self.build_google_news_url(DEFAULT_GOOGLE_NEWS_QUERY, when=when)
        return await self.fetch_feed(feed_url, limit=limit)

    async def fetch_topic_news(
        self,
        topic: str,
        *,
        limit: int = 10,
        when: str = "7d",
    ) -> list[NewsArticle]:
        query = topic.strip()
        if not query:
            return []
        feed_url = self.build_google_news_url(query, when=when)
        return await self.fetch_feed(feed_url, limit=limit)

    async def fetch_feed(self, feed_url: str, *, limit: int = 10) -> list[NewsArticle]:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = await client.get(feed_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch news feed {}: {}", feed_url, exc)
            return []

        try:
            return self._parse_rss(response.text, limit=limit)
        except Exception as exc:
            logger.exception("Failed to parse news feed {}: {}", feed_url, exc)
            return []

    @staticmethod
    def build_google_news_url(
        query: str,
        *,
        when: str = "1d",
        hl: str = "en-US",
        gl: str = "US",
        ceid: str = "US:en",
    ) -> str:
        effective_query = query.strip()
        if when.strip():
            effective_query = f"{effective_query} when:{when.strip()}"
        return f"https://news.google.com/rss/search?q={quote_plus(effective_query)}&hl={hl}&gl={gl}&ceid={ceid}"

    def _parse_rss(self, xml_payload: str, *, limit: int) -> list[NewsArticle]:
        root = ET.fromstring(xml_payload)
        items = root.findall("./channel/item")
        articles: list[NewsArticle] = []

        for item in items[: max(limit, 0)]:
            try:
                title = self._clean_text(item.findtext("title"))
                link = (item.findtext("link") or "").strip()
                guid = self._clean_text(item.findtext("guid"))
                description = self._strip_html(item.findtext("description"))
                published_at = self._parse_pub_date(item.findtext("pubDate"))

                source_node = item.find("source")
                source = self._clean_text(source_node.text if source_node is not None else None)
                if source is None:
                    source = self._extract_source_from_title(title)

                if not title or not link:
                    continue

                articles.append(
                    NewsArticle(
                        title=title,
                        link=link,
                        source=source,
                        published_at=published_at,
                        summary=description,
                        guid=guid,
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed RSS item: {}", exc)

        return articles

    @staticmethod
    def _parse_pub_date(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            timestamp = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    @staticmethod
    def _clean_text(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = html.unescape(value).strip()
        return normalized or None

    @staticmethod
    def _strip_html(value: str | None) -> str | None:
        if value is None:
            return None
        text = re.sub(r"<[^>]+>", " ", value)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    @staticmethod
    def _extract_source_from_title(title: str | None) -> str | None:
        if not title:
            return None
        if " - " not in title:
            return None
        source = title.rsplit(" - ", 1)[-1].strip()
        return source or None

    async def fetch_from_feeds(
        self,
        feed_urls: Iterable[str],
        *,
        limit_per_feed: int = 5,
    ) -> list[NewsArticle]:
        feed_url_list = list(feed_urls)
        tasks = [self.fetch_feed(feed_url, limit=limit_per_feed) for feed_url in feed_url_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[NewsArticle] = []
        for feed_url, result in zip(feed_url_list, results, strict=False):
            if isinstance(result, Exception):
                logger.exception("Feed task failed for {}: {}", feed_url, result)
                continue
            articles.extend(result)
        return articles

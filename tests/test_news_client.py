from __future__ import annotations

import pytest

from invest_advisor_bot.providers.news_client import NewsClient


RSS_PAYLOAD = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Fed holds rates steady</title>
      <link>https://example.com/fed</link>
      <guid>1</guid>
      <pubDate>Fri, 01 Mar 2024 10:00:00 GMT</pubDate>
      <description><![CDATA[<p>Rates unchanged</p>]]></description>
      <source>Reuters</source>
    </item>
  </channel>
</rss>
"""


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        return _FakeResponse(RSS_PAYLOAD)


@pytest.mark.asyncio
async def test_news_client_fetch_feed_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    client = NewsClient(cache_ttl_seconds=900)
    monkeypatch.setattr("invest_advisor_bot.providers.news_client.httpx.AsyncClient", _FakeAsyncClient)

    first = await client.fetch_feed("https://example.com/feed", limit=3)
    second = await client.fetch_feed("https://example.com/feed", limit=3)

    assert len(first) == 1
    assert first[0].title == "Fed holds rates steady"
    assert second[0].source == "Reuters"

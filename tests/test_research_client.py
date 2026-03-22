from __future__ import annotations

import httpx
import pytest

from invest_advisor_bot.providers import research_client as research_mod


class _MockResponse:
    def __init__(self, *, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.com")
            raise httpx.HTTPStatusError("error", request=request, response=self)

    def json(self) -> dict:
        return self._payload


class _MockAsyncClient:
    def __init__(self, *, queue: list[_MockResponse], calls: list[dict], **_: object) -> None:
        self._queue = queue
        self._calls = calls

    async def __aenter__(self) -> "_MockAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, headers: dict | None = None, json: dict | None = None) -> _MockResponse:
        self._calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        return self._queue.pop(0)


@pytest.mark.asyncio
async def test_research_client_merges_tavily_and_exa(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = [
        _MockResponse(
            status_code=200,
            payload={
                "results": [
                    {
                        "title": "Fed signals caution",
                        "url": "https://example.com/fed",
                        "source": "Reuters",
                        "content": "Fed keeps stance cautious.",
                        "score": 0.91,
                    }
                ]
            },
        ),
        _MockResponse(
            status_code=200,
            payload={
                "results": [
                    {
                        "title": "Gold demand rises",
                        "url": "https://example.com/gold",
                        "summary": "Gold gains on safe-haven demand.",
                        "publishedDate": "2026-03-18T12:00:00Z",
                    }
                ]
            },
        ),
    ]
    calls: list[dict] = []
    monkeypatch.setattr(
        research_mod.httpx,
        "AsyncClient",
        lambda **kwargs: _MockAsyncClient(queue=queue, calls=calls, **kwargs),
    )

    client = research_mod.ResearchClient(
        tavily_api_key="tvly-key",
        exa_api_key="exa-key",
        provider_order=("tavily", "exa"),
        cache_ttl_seconds=60,
    )
    findings = await client.search_market_context(query="gold and fed", limit=4)

    assert len(findings) == 2
    assert findings[0].provider == "tavily"
    assert findings[1].provider == "exa"
    assert calls[0]["url"] == "https://api.tavily.com/search"
    assert calls[1]["url"] == "https://api.exa.ai/search"


@pytest.mark.asyncio
async def test_research_client_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = [
        _MockResponse(
            status_code=200,
            payload={
                "results": [
                    {
                        "title": "ETF inflows steady",
                        "url": "https://example.com/etf",
                        "source": "Bloomberg",
                        "content": "Inflows remain healthy.",
                    }
                ]
            },
        )
    ]
    calls: list[dict] = []
    monkeypatch.setattr(
        research_mod.httpx,
        "AsyncClient",
        lambda **kwargs: _MockAsyncClient(queue=queue, calls=calls, **kwargs),
    )

    client = research_mod.ResearchClient(
        tavily_api_key="tvly-key",
        exa_api_key="",
        provider_order=("tavily",),
        cache_ttl_seconds=60,
    )

    first = await client.search_market_context(query="ETF inflows", limit=3)
    second = await client.search_market_context(query="ETF inflows", limit=3)

    assert len(first) == 1
    assert len(second) == 1
    assert len(calls) == 1

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import RLock
from typing import Any, Mapping, Sequence

import httpx
from cachetools import TTLCache
from loguru import logger


DEFAULT_RESEARCH_PROVIDER_ORDER: tuple[str, ...] = ("tavily", "exa")


@dataclass(slots=True, frozen=True)
class ResearchFinding:
    title: str
    url: str
    source: str
    snippet: str
    provider: str
    published_at: datetime | None = None
    score: float | None = None


class ResearchClient:
    """Async web research client that can pull fresh market context from Tavily and Exa."""

    def __init__(
        self,
        *,
        tavily_api_key: str = "",
        exa_api_key: str = "",
        provider_order: Sequence[str] = DEFAULT_RESEARCH_PROVIDER_ORDER,
        timeout: float = 12.0,
        cache_ttl_seconds: int = 900,
        cache_maxsize: int = 128,
    ) -> None:
        self.tavily_api_key = tavily_api_key.strip()
        self.exa_api_key = exa_api_key.strip()
        self.provider_order = tuple(
            dict.fromkeys(item.strip().casefold() for item in provider_order if item and item.strip())
        ) or DEFAULT_RESEARCH_PROVIDER_ORDER
        self.timeout = timeout
        self._http_client: httpx.AsyncClient | None = None
        self._cache_lock = RLock()
        self._search_cache: TTLCache[tuple[str, int, tuple[str, ...]], list[ResearchFinding]] = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )

    def available(self) -> bool:
        return bool(self.tavily_api_key or self.exa_api_key)

    async def search_market_context(self, *, query: str, limit: int = 5) -> list[ResearchFinding]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        cache_key = (normalized_query, max(1, limit), self.provider_order)
        with self._cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                return list(cached)

        tasks = []
        if self.tavily_api_key and "tavily" in self.provider_order:
            tasks.append(self._search_tavily(query=normalized_query, limit=limit))
        if self.exa_api_key and "exa" in self.provider_order:
            tasks.append(self._search_exa(query=normalized_query, limit=limit))
        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        findings: list[ResearchFinding] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Research task failed: {}", result)
                continue
            findings.extend(result)

        deduped = self._dedupe_findings(findings, limit=limit)
        with self._cache_lock:
            self._search_cache[cache_key] = list(deduped)
        return deduped

    async def search_earnings_call_context(
        self,
        *,
        ticker: str,
        company_name: str | None = None,
        limit: int = 4,
    ) -> list[ResearchFinding]:
        symbol = ticker.strip().upper()
        if not symbol:
            return []
        name_part = f' OR "{company_name.strip()}"' if company_name and company_name.strip() else ""
        query = (
            f'"{symbol}"{name_part} '
            '("earnings call" OR transcript OR "conference call" OR guidance OR outlook OR commentary)'
        )
        return await self.search_market_context(query=query, limit=limit)

    async def _search_tavily(self, *, query: str, limit: int) -> list[ResearchFinding]:
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "topic": "news",
            "max_results": max(2, limit),
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            client = self._get_http_client()
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Tavily search failed: {}", exc)
            return []

        findings: list[ResearchFinding] = []
        for item in data.get("results", [])[:limit]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            findings.append(
                ResearchFinding(
                    title=title,
                    url=url,
                    source=str(item.get("source") or "Tavily"),
                    snippet=str(item.get("content") or item.get("snippet") or "").strip(),
                    provider="tavily",
                    published_at=self._parse_datetime(item.get("published_date")),
                    score=self._as_float(item.get("score")),
                )
            )
        return findings

    async def _search_exa(self, *, query: str, limit: int) -> list[ResearchFinding]:
        payload = {
            "query": query,
            "numResults": max(2, limit),
            "type": "auto",
            "contents": {
                "summary": {"query": "Summarize the market relevance in one short paragraph."},
                "highlights": {"numSentences": 2},
            },
        }
        headers = {
            "x-api-key": self.exa_api_key,
            "Content-Type": "application/json",
        }
        try:
            client = self._get_http_client()
            response = await client.post("https://api.exa.ai/search", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Exa search failed: {}", exc)
            return []

        findings: list[ResearchFinding] = []
        for item in data.get("results", [])[:limit]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            snippet = self._coalesce_exa_snippet(item)
            findings.append(
                ResearchFinding(
                    title=title,
                    url=url,
                    source=self._extract_source(item.get("url")),
                    snippet=snippet,
                    provider="exa",
                    published_at=self._parse_datetime(item.get("publishedDate")),
                    score=None,
                )
            )
        return findings

    @staticmethod
    def _coalesce_exa_snippet(item: Mapping[str, Any]) -> str:
        summary = item.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        highlights = item.get("highlights")
        if isinstance(highlights, list):
            parts = [str(part).strip() for part in highlights if str(part).strip()]
            if parts:
                return " ".join(parts)
        text = item.get("text")
        if isinstance(text, str):
            return text.strip()[:500]
        return ""

    @staticmethod
    def _dedupe_findings(findings: Sequence[ResearchFinding], *, limit: int) -> list[ResearchFinding]:
        deduped: list[ResearchFinding] = []
        seen_urls: set[str] = set()
        for finding in findings:
            normalized_url = finding.url.strip().lower()
            if not normalized_url or normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            deduped.append(finding)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _extract_source(url: object) -> str:
        value = str(url or "").strip()
        if not value:
            return "Exa"
        try:
            return value.split("//", 1)[-1].split("/", 1)[0]
        except Exception:
            return "Exa"

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                return None
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _as_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def aclose(self) -> None:
        client = self._http_client
        self._http_client = None
        if client is not None:
            await client.aclose()

    def _get_http_client(self) -> httpx.AsyncClient:
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            self._http_client = client
        return client

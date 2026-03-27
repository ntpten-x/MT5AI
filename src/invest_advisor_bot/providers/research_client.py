from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from threading import RLock
from typing import Any, Mapping, Sequence

import httpx
from cachetools import TTLCache
from loguru import logger

from invest_advisor_bot.runtime_diagnostics import diagnostics


DEFAULT_RESEARCH_PROVIDER_ORDER: tuple[str, ...] = ("tavily", "exa")
_TRANSCRIPT_PAGE_HINTS: tuple[str, ...] = (
    "transcript",
    "earnings call",
    "conference call",
    "prepared remarks",
    "operator",
    "question-and-answer",
    "guidance",
    "outlook",
    "investor relations",
)


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
        self.cache_ttl_seconds = cache_ttl_seconds
        self._http_client: httpx.AsyncClient | None = None
        self._cache_lock = RLock()
        self._search_cache: TTLCache[tuple[str, int, tuple[str, ...]], list[ResearchFinding]] = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )
        self._page_cache: TTLCache[str, str | None] = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl_seconds)

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
        findings = await self.search_market_context(query=query, limit=limit)
        return await self._enrich_earnings_call_findings(findings)

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
        started_at = asyncio.get_running_loop().time()
        try:
            client = self._get_http_client()
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            diagnostics.record_provider_latency(
                service="research_client",
                provider="tavily",
                operation="search_market_context",
                latency_ms=(asyncio.get_running_loop().time() - started_at) * 1000.0,
                success=False,
            )
            logger.warning("Tavily search failed: {}", exc)
            return []
        diagnostics.record_provider_latency(
            service="research_client",
            provider="tavily",
            operation="search_market_context",
            latency_ms=(asyncio.get_running_loop().time() - started_at) * 1000.0,
            success=True,
        )

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
        started_at = asyncio.get_running_loop().time()
        try:
            client = self._get_http_client()
            response = await client.post("https://api.exa.ai/search", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            diagnostics.record_provider_latency(
                service="research_client",
                provider="exa",
                operation="search_market_context",
                latency_ms=(asyncio.get_running_loop().time() - started_at) * 1000.0,
                success=False,
            )
            logger.warning("Exa search failed: {}", exc)
            return []
        diagnostics.record_provider_latency(
            service="research_client",
            provider="exa",
            operation="search_market_context",
            latency_ms=(asyncio.get_running_loop().time() - started_at) * 1000.0,
            success=True,
        )

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

    async def _enrich_earnings_call_findings(self, findings: Sequence[ResearchFinding]) -> list[ResearchFinding]:
        if not findings:
            return []
        enriched: list[ResearchFinding] = []
        for finding in findings:
            transcript_text = await self._fetch_transcript_page_text(finding.url)
            if not transcript_text:
                enriched.append(finding)
                continue
            snippet = self._extract_transcript_like_snippet(transcript_text)
            if not snippet:
                enriched.append(finding)
                continue
            enriched.append(
                ResearchFinding(
                    title=finding.title,
                    url=finding.url,
                    source=finding.source,
                    snippet=snippet,
                    provider=finding.provider,
                    published_at=finding.published_at,
                    score=finding.score,
                )
            )
        return enriched

    async def _fetch_transcript_page_text(self, url: str) -> str | None:
        normalized_url = str(url or "").strip()
        if not normalized_url:
            return None
        with self._cache_lock:
            if normalized_url in self._page_cache:
                return self._page_cache[normalized_url]
        try:
            client = self._get_http_client()
            response = await client.get(
                normalized_url,
                headers={
                    "User-Agent": "invest-advisor-bot/0.2",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                follow_redirects=True,
            )
            response.raise_for_status()
            content_type = str(response.headers.get("content-type") or "").casefold()
            body = response.text
        except httpx.HTTPError as exc:
            logger.debug("Transcript page fetch failed for {}: {}", normalized_url, exc)
            body = ""
            content_type = ""
        text: str | None = None
        if "html" in content_type or body.lstrip().startswith("<"):
            text = self._html_to_text(body)
        elif body:
            text = self._normalize_text(body)
        if text and not self._looks_like_transcript_page(text, normalized_url):
            text = None
        with self._cache_lock:
            self._page_cache[normalized_url] = text
        return text

    @classmethod
    def _extract_transcript_like_snippet(cls, text: str) -> str:
        normalized = cls._normalize_text(text)
        if not normalized:
            return ""
        lowered = normalized.casefold()
        start_index = 0
        for hint in ("prepared remarks", "guidance", "outlook", "question-and-answer", "operator"):
            candidate_index = lowered.find(hint)
            if candidate_index >= 0:
                start_index = candidate_index
                break
        excerpt = normalized[start_index : start_index + 900]
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", excerpt) if part.strip()]
        selected: list[str] = []
        for sentence in sentences:
            lowered_sentence = sentence.casefold()
            if any(keyword in lowered_sentence for keyword in ("guidance", "outlook", "demand", "margin", "inventory", "pricing", "pipeline", "forecast")):
                selected.append(sentence)
            if len(selected) >= 4:
                break
        return " ".join(selected[:4])[:700] if selected else excerpt[:700]

    @classmethod
    def _looks_like_transcript_page(cls, text: str, url: str) -> bool:
        lowered_text = text.casefold()
        lowered_url = str(url or "").casefold()
        return any(token in lowered_text or token in lowered_url for token in _TRANSCRIPT_PAGE_HINTS)

    @classmethod
    def _html_to_text(cls, html: str) -> str:
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?i)<br\\s*/?>", "\n", text)
        text = re.sub(r"(?i)</p>|</div>|</section>|</article>|</li>|</tr>|</h\\d>", "\n", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return cls._normalize_text(unescape(text))

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

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

    def status(self) -> dict[str, object]:
        with self._cache_lock:
            cached_queries = len(self._search_cache)
            cached_pages = len(self._page_cache)
        configured_providers = {
            "tavily": bool(self.tavily_api_key),
            "exa": bool(self.exa_api_key),
        }
        return {
            "available": self.available(),
            "provider_order": list(self.provider_order),
            "configured_providers": configured_providers,
            "timeout_seconds": self.timeout,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cached_queries": cached_queries,
            "cached_pages": cached_pages,
        }

    def _get_http_client(self) -> httpx.AsyncClient:
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
            self._http_client = client
        return client

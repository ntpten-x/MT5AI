from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from threading import RLock
from typing import Any

import httpx
from cachetools import TTLCache
from loguru import logger


@dataclass(slots=True, frozen=True)
class PolicyFeedEvent:
    central_bank: str
    title: str
    category: str
    published_at: datetime | None
    url: str | None
    summary: str | None
    tone_signal: str | None = None


class PolicyFeedClient:
    def __init__(
        self,
        *,
        enabled: bool = True,
        fed_speeches_feed_url: str = "https://www.federalreserve.gov/feeds/speeches.xml",
        fed_press_feed_url: str = "https://www.federalreserve.gov/feeds/press_all.xml",
        ecb_press_feed_url: str = "https://www.ecb.europa.eu/rss/press.html",
        ecb_speeches_feed_url: str = "https://www.ecb.europa.eu/rss/speeches.html",
        timeout_seconds: float = 12.0,
        cache_ttl_seconds: int = 900,
    ) -> None:
        self.enabled = bool(enabled)
        self.fed_speeches_feed_url = fed_speeches_feed_url.strip()
        self.fed_press_feed_url = fed_press_feed_url.strip()
        self.ecb_press_feed_url = ecb_press_feed_url.strip()
        self.ecb_speeches_feed_url = ecb_speeches_feed_url.strip()
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self._cache = TTLCache(maxsize=8, ttl=max(60, int(cache_ttl_seconds)))
        self._lock = RLock()
        self._http_client: httpx.Client | None = None
        self._warning: str | None = None

    async def aclose(self) -> None:
        with self._lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            client.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": True,
                "enabled": self.enabled,
                "warning": self._warning,
                "cache_entries": len(self._cache),
                "fed_speeches_feed_url": self.fed_speeches_feed_url or None,
                "ecb_press_feed_url": self.ecb_press_feed_url or None,
            }

    async def fetch_recent_policy_events(self, *, limit: int = 6) -> list[PolicyFeedEvent]:
        if not self.enabled:
            return []
        cache_key = f"policy:{max(1, limit)}"
        with self._lock:
            cached = self._cache.get(cache_key)
        if isinstance(cached, list):
            return list(cached)
        result = await asyncio.to_thread(self._fetch_recent_policy_events_sync, max(1, limit))
        with self._lock:
            self._cache[cache_key] = list(result)
        return result

    def _fetch_recent_policy_events_sync(self, limit: int) -> list[PolicyFeedEvent]:
        events: list[PolicyFeedEvent] = []
        for central_bank, category, url in (
            ("fed", "speech", self.fed_speeches_feed_url),
            ("fed", "press", self.fed_press_feed_url),
            ("ecb", "press", self.ecb_press_feed_url),
            ("ecb", "speech", self.ecb_speeches_feed_url),
        ):
            if not url:
                continue
            text = self._fetch_text(url)
            if not text:
                continue
            events.extend(self._parse_feed(text=text, central_bank=central_bank, category=category))
        events.sort(key=lambda item: item.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return events[:limit]

    def _fetch_text(self, url: str) -> str | None:
        try:
            response = self._get_http_client().get(
                url,
                headers={"User-Agent": "InvestAdvisorBot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            with self._lock:
                self._warning = f"policy_feed_fetch_failed: {exc}"
            logger.warning("Policy feed fetch failed for {}: {}", url, exc)
            return None

    def _parse_feed(self, *, text: str, central_bank: str, category: str) -> list[PolicyFeedEvent]:
        normalized = text.strip()
        if not normalized:
            return []
        if normalized.startswith("<"):
            try:
                root = ET.fromstring(normalized)
            except ET.ParseError:
                return self._parse_html_links(text=normalized, central_bank=central_bank, category=category)
            items = root.findall(".//item") or root.findall(".//entry")
            parsed: list[PolicyFeedEvent] = []
            for item in items[:12]:
                title = self._extract_child_text(item, ("title",))
                link = self._extract_child_text(item, ("link", "id"))
                summary = self._extract_child_text(item, ("description", "summary"))
                published = self._parse_datetime(
                    self._extract_child_text(item, ("pubDate", "updated", "published", "dc:date"))
                )
                if title:
                    parsed.append(
                        PolicyFeedEvent(
                            central_bank=central_bank,
                            title=title,
                            category=category,
                            published_at=published,
                            url=link,
                            summary=summary,
                            tone_signal=self._classify_tone(title=title, summary=summary),
                        )
                    )
            return parsed
        return self._parse_html_links(text=normalized, central_bank=central_bank, category=category)

    def _parse_html_links(self, *, text: str, central_bank: str, category: str) -> list[PolicyFeedEvent]:
        matches = re.findall(r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', text, flags=re.I | re.S)
        parsed: list[PolicyFeedEvent] = []
        for href, title_html in matches[:12]:
            title = re.sub(r"(?is)<[^>]+>", " ", unescape(title_html or ""))
            title = re.sub(r"\s+", " ", title).strip()
            if not title:
                continue
            parsed.append(
                PolicyFeedEvent(
                    central_bank=central_bank,
                    title=title,
                    category=category,
                    published_at=None,
                    url=href,
                    summary=None,
                    tone_signal=self._classify_tone(title=title, summary=None),
                )
            )
        return parsed

    @staticmethod
    def _extract_child_text(node: ET.Element, tags: tuple[str, ...]) -> str | None:
        for child in list(node):
            tag_name = child.tag.split("}")[-1]
            if tag_name in tags:
                if tag_name == "link" and child.attrib.get("href"):
                    return child.attrib.get("href")
                text = str(child.text or "").strip()
                if text:
                    return text
        return None

    @staticmethod
    def _classify_tone(*, title: str | None, summary: str | None) -> str | None:
        text = f"{title or ''} {summary or ''}".casefold()
        if any(token in text for token in ("inflation", "tight", "restrictive", "higher for longer", "price stability")):
            return "hawkish"
        if any(token in text for token in ("growth slowdown", "support", "easing", "employment risk", "downside risk")):
            return "dovish"
        if text.strip():
            return "neutral"
        return None

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None

    def _get_http_client(self) -> httpx.Client:
        with self._lock:
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.timeout_seconds)
            return self._http_client

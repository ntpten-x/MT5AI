from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

import httpx
from cachetools import TTLCache
from loguru import logger

from invest_advisor_bot.runtime_diagnostics import diagnostics


@dataclass(slots=True, frozen=True)
class TranscriptInsight:
    ticker: str
    quarter: int | None
    year: int | None
    published_at: datetime | None
    source: str
    tone: str
    guidance_signal: str
    confidence: float
    summary: str
    highlights: tuple[str, ...]


class EarningsTranscriptClient:
    """Fetch earnings call transcripts and derive management commentary signals."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://financialmodelingprep.com/api/v3",
        alpha_vantage_api_key: str = "",
        alpha_vantage_base_url: str = "https://www.alphavantage.co/query",
        timeout_seconds: float = 12.0,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/") or "https://financialmodelingprep.com/api/v3"
        self.alpha_vantage_api_key = alpha_vantage_api_key.strip()
        self.alpha_vantage_base_url = (
            alpha_vantage_base_url.strip().rstrip("/") or "https://www.alphavantage.co/query"
        )
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self._http_client: httpx.Client | None = None
        self._lock = RLock()
        self._cache: TTLCache[str, TranscriptInsight | None] = TTLCache(maxsize=128, ttl=max(300, cache_ttl_seconds))
        self._warning: str | None = None
        self._active_backend: str | None = None
        self._provider_warnings: dict[str, str | None] = {
            "financial_modeling_prep": None,
            "alpha_vantage": None,
        }
        self._provider_disabled: dict[str, bool] = {
            "financial_modeling_prep": False,
            "alpha_vantage": False,
        }

    def available(self) -> bool:
        return any(self._provider_available(provider) for provider in self._provider_order())

    def status(self) -> dict[str, Any]:
        provider_statuses = {
            provider: {
                "configured": self._provider_configured(provider),
                "disabled": self._provider_disabled.get(provider, False),
                "warning": self._provider_warnings.get(provider),
            }
            for provider in self._provider_order()
        }
        return {
            "available": self.available(),
            "backend": self._active_backend or self._default_backend_label(),
            "mode": "direct_api_with_fallback" if len(self._provider_order()) > 1 else "direct_api",
            "disabled": not self.available(),
            "base_url": self.base_url,
            "alpha_vantage_base_url": self.alpha_vantage_base_url,
            "provider_order": self._provider_order(),
            "provider_statuses": provider_statuses,
            "warning": self._warning,
            "cache_entries": len(self._cache),
        }

    async def get_latest_management_commentary(
        self,
        ticker: str,
        *,
        max_quarters_back: int = 8,
    ) -> TranscriptInsight | None:
        normalized = ticker.strip().upper()
        if not normalized:
            return None
        with self._lock:
            if normalized in self._cache:
                return self._cache[normalized]
        result = await asyncio.to_thread(self._get_latest_management_commentary_sync, normalized, max_quarters_back)
        with self._lock:
            self._cache[normalized] = result
        return result

    async def get_latest_management_commentary_batch(
        self,
        tickers: Sequence[str],
        *,
        limit: int = 4,
    ) -> dict[str, TranscriptInsight]:
        unique = []
        for ticker in tickers:
            normalized = ticker.strip().upper()
            if normalized and normalized not in unique:
                unique.append(normalized)
            if len(unique) >= max(1, limit):
                break
        if not unique:
            return {}
        tasks = {ticker: self.get_latest_management_commentary(ticker) for ticker in unique}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, TranscriptInsight] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            if isinstance(result, Exception) or result is None:
                continue
            payload[ticker] = result
        return payload

    def _get_latest_management_commentary_sync(self, ticker: str, max_quarters_back: int) -> TranscriptInsight | None:
        if not self.available():
            self._warning = "transcript client disabled or missing API key"
            return None
        self._warning = None
        for provider in self._provider_order():
            if not self._provider_available(provider):
                continue
            started_at = self._monotonic()
            transcript_payload = self._fetch_provider_transcript_payload(
                provider,
                ticker=ticker,
                max_quarters_back=max_quarters_back,
            )
            diagnostics.record_provider_latency(
                service="transcript_client",
                provider=provider,
                operation="earnings_transcript",
                latency_ms=(self._monotonic() - started_at) * 1000.0,
                success=transcript_payload is not None,
            )
            if transcript_payload is None:
                continue
            self._active_backend = provider
            self._warning = self._provider_warnings.get(provider)
            return self._build_insight(ticker=ticker, payload=transcript_payload)
        self._warning = next(
            (
                warning
                for provider in self._provider_order()
                for warning in [self._provider_warnings.get(provider)]
                if warning
            ),
            "transcript endpoints unavailable; using research fallback when needed",
        )
        return None

    def _fetch_provider_transcript_payload(
        self,
        provider: str,
        *,
        ticker: str,
        max_quarters_back: int,
    ) -> Mapping[str, Any] | None:
        if provider == "financial_modeling_prep":
            now = datetime.now(timezone.utc)
            for offset in range(max(1, max_quarters_back)):
                quarter_index = ((now.month - 1) // 3) + 1 - offset
                year = now.year
                while quarter_index <= 0:
                    quarter_index += 4
                    year -= 1
                payload = self._fetch_fmp_transcript_payload(ticker, quarter=quarter_index, year=year)
                if payload:
                    return payload
                if self._provider_disabled.get(provider):
                    break
            return None
        if provider == "alpha_vantage":
            return self._fetch_alpha_vantage_transcript_payload(ticker)
        return None

    def _fetch_fmp_transcript_payload(self, ticker: str, *, quarter: int, year: int) -> Mapping[str, Any] | None:
        try:
            response = self._get_http_client().get(
                f"{self.base_url}/earning_call_transcript/{ticker}",
                params={"quarter": quarter, "year": year, "apikey": self.api_key},
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code in {402, 403}:
                self._provider_disabled["financial_modeling_prep"] = True
                if status_code == 402:
                    warning = "transcript endpoint restricted for current FMP subscription"
                else:
                    warning = "transcript endpoint forbidden for current FMP plan or key"
                self._provider_warnings["financial_modeling_prep"] = warning
                self._warning = warning
                logger.warning(
                    "Transcript endpoint disabled after HTTP {} for {} Q{} {}",
                    status_code,
                    ticker,
                    quarter,
                    year,
                )
                return None
            logger.warning("Transcript request failed for {} Q{} {}: {}", ticker, quarter, year, exc)
            self._provider_warnings["financial_modeling_prep"] = str(exc)
            self._warning = str(exc)
            return None
        except Exception as exc:
            logger.warning("Transcript request failed for {} Q{} {}: {}", ticker, quarter, year, exc)
            self._provider_warnings["financial_modeling_prep"] = str(exc)
            self._warning = str(exc)
            return None
        if isinstance(payload, list):
            first = next((item for item in payload if isinstance(item, Mapping)), None)
            if isinstance(first, Mapping):
                item = dict(first)
                item.setdefault("_provider", "financial_modeling_prep")
                return item
            return None
        if isinstance(payload, Mapping):
            item = dict(payload)
            item.setdefault("_provider", "financial_modeling_prep")
            return item
        return None

    def _fetch_alpha_vantage_transcript_payload(self, ticker: str) -> Mapping[str, Any] | None:
        try:
            response = self._get_http_client().get(
                self.alpha_vantage_base_url,
                params={
                    "function": "EARNINGS_CALL_TRANSCRIPT",
                    "symbol": ticker,
                    "apikey": self.alpha_vantage_api_key,
                },
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Alpha Vantage transcript request failed for {}: {}", ticker, exc)
            self._provider_warnings["alpha_vantage"] = str(exc)
            self._warning = str(exc)
            return None
        except Exception as exc:
            logger.warning("Alpha Vantage transcript request failed for {}: {}", ticker, exc)
            self._provider_warnings["alpha_vantage"] = str(exc)
            self._warning = str(exc)
            return None
        if isinstance(payload, Mapping):
            note = str(payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "").strip()
            if note:
                self._handle_alpha_vantage_note(note)
                return None
            nested_payload = payload.get("data")
            if isinstance(nested_payload, list):
                payload = nested_payload
            elif any(payload.get(key) for key in ("transcript", "content", "text")):
                item = dict(payload)
                item.setdefault("_provider", "alpha_vantage")
                return item
        if isinstance(payload, list):
            first = next((item for item in payload if isinstance(item, Mapping)), None)
            if isinstance(first, Mapping):
                item = dict(first)
                item.setdefault("_provider", "alpha_vantage")
                return item
        return None

    def _build_insight(self, *, ticker: str, payload: Mapping[str, Any]) -> TranscriptInsight | None:
        content = str(payload.get("content") or payload.get("transcript") or payload.get("text") or "").strip()
        if not content:
            return None
        return self._build_text_insight(
            ticker=ticker,
            source=str(payload.get("_provider") or payload.get("symbol") or payload.get("ticker") or "transcript").strip(),
            content=content,
            published_at=self._parse_datetime(
                payload.get("date") or payload.get("published_at") or payload.get("timestamp")
            ),
            quarter=self._as_int(payload.get("quarter")),
            year=self._as_int(payload.get("year")),
        )

    @classmethod
    def build_research_proxy_insight(
        cls,
        *,
        ticker: str,
        findings: Sequence[Mapping[str, Any] | Any],
    ) -> TranscriptInsight | None:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return None
        text_fragments: list[str] = []
        published_candidates: list[datetime] = []
        provider_names: list[str] = []
        for finding in findings:
            if not isinstance(finding, Mapping):
                title = str(getattr(finding, "title", "") or "").strip()
                snippet = str(getattr(finding, "snippet", "") or "").strip()
                provider = str(getattr(finding, "provider", "") or "").strip()
                published_at = getattr(finding, "published_at", None)
            else:
                title = str(finding.get("title") or "").strip()
                snippet = str(finding.get("snippet") or "").strip()
                provider = str(finding.get("provider") or "").strip()
                published_at = finding.get("published_at")
            fragment = " ".join(part for part in (title, snippet) if part).strip()
            if fragment:
                text_fragments.append(fragment)
            if provider and provider not in provider_names:
                provider_names.append(provider)
            parsed_published_at = published_at if isinstance(published_at, datetime) else cls._parse_datetime(published_at)
            if parsed_published_at is not None:
                published_candidates.append(parsed_published_at)
        if not text_fragments:
            return None
        source = "research_proxy"
        if provider_names:
            source = f"research_proxy:{'+'.join(provider_names[:2])}"
        return cls._build_text_insight(
            ticker=normalized_ticker,
            source=source,
            content=" ".join(text_fragments),
            published_at=max(published_candidates) if published_candidates else None,
            quarter=None,
            year=None,
        )

    @classmethod
    def _build_text_insight(
        cls,
        *,
        ticker: str,
        source: str,
        content: str,
        published_at: datetime | None,
        quarter: int | None,
        year: int | None,
    ) -> TranscriptInsight | None:
        sentences = [cls._normalize_sentence(sentence) for sentence in re.split(r"(?<=[.!?])\s+", content) if sentence.strip()]
        positive_tokens = ("demand", "strong", "accelerat", "margin expansion", "backlog", "healthy", "improv", "outperform")
        negative_tokens = ("headwind", "soft", "uncertain", "pressure", "slow", "inventory", "weaker", "challenging")
        guidance_positive_tokens = ("raise guidance", "above expectations", "confident", "reaffirm", "strong pipeline")
        guidance_negative_tokens = ("lower guidance", "cautious", "visibility", "macro uncertainty", "cost pressure")
        positive_hits = sum(1 for sentence in sentences for token in positive_tokens if token in sentence.casefold())
        negative_hits = sum(1 for sentence in sentences for token in negative_tokens if token in sentence.casefold())
        guidance_positive_hits = sum(1 for sentence in sentences for token in guidance_positive_tokens if token in sentence.casefold())
        guidance_negative_hits = sum(1 for sentence in sentences for token in guidance_negative_tokens if token in sentence.casefold())
        tone_score = positive_hits - negative_hits
        guidance_score = guidance_positive_hits - guidance_negative_hits
        tone = "balanced"
        if tone_score >= 2:
            tone = "constructive"
        elif tone_score <= -2:
            tone = "cautious"
        guidance_signal = "mixed"
        if guidance_score >= 1:
            guidance_signal = "supportive"
        elif guidance_score <= -1:
            guidance_signal = "softening"
        confidence = min(0.95, max(0.35, 0.5 + (abs(tone_score) * 0.08) + (abs(guidance_score) * 0.1)))
        highlights = cls._select_highlights(sentences)
        summary = (
            f"{ticker} management tone {tone}, guidance {guidance_signal}, "
            f"positive_hits={positive_hits}, negative_hits={negative_hits}"
        )
        return TranscriptInsight(
            ticker=ticker,
            quarter=quarter,
            year=year,
            published_at=published_at,
            source=source,
            tone=tone,
            guidance_signal=guidance_signal,
            confidence=round(confidence, 2),
            summary=summary,
            highlights=tuple(highlights[:4]),
        )

    def _get_http_client(self) -> httpx.Client:
        with self._lock:
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.timeout_seconds)
            return self._http_client

    def _provider_order(self) -> list[str]:
        order: list[str] = []
        if self.api_key:
            order.append("financial_modeling_prep")
        if self.alpha_vantage_api_key:
            order.append("alpha_vantage")
        return order

    def _provider_configured(self, provider: str) -> bool:
        if provider == "financial_modeling_prep":
            return bool(self.api_key)
        if provider == "alpha_vantage":
            return bool(self.alpha_vantage_api_key)
        return False

    def _provider_available(self, provider: str) -> bool:
        return self._provider_configured(provider) and not self._provider_disabled.get(provider, False)

    def _default_backend_label(self) -> str:
        order = self._provider_order()
        if not order:
            return "unconfigured"
        if len(order) == 1:
            return order[0]
        return "multi_source"

    def _handle_alpha_vantage_note(self, note: str) -> None:
        normalized = note.casefold()
        if "25 requests per day" in normalized or "daily rate limit" in normalized:
            self._provider_disabled["alpha_vantage"] = True
            warning = "alpha vantage transcript disabled for current runtime after free daily-limit response"
        elif "rate limit" in normalized or "1 request per minute" in normalized:
            warning = "alpha vantage transcript rate limited on free tier"
        else:
            warning = note[:240]
        self._provider_warnings["alpha_vantage"] = warning
        self._warning = warning

    @staticmethod
    def _normalize_sentence(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @staticmethod
    def _select_highlights(sentences: Sequence[str]) -> list[str]:
        keywords = ("guidance", "demand", "margin", "pricing", "inventory", "AI", "macro", "backlog", "pipeline")
        highlights: list[str] = []
        for sentence in sentences:
            normalized = sentence.casefold()
            if any(keyword.casefold() in normalized for keyword in keywords):
                highlights.append(sentence[:220])
            if len(highlights) >= 4:
                break
        return highlights or [sentence[:220] for sentence in sentences[:2]]

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _monotonic() -> float:
        import time

        return time.perf_counter()

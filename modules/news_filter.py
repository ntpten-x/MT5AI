from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import requests
from loguru import logger

from config import Settings

IMPACT_SCORES = {"low": 1, "medium": 2, "high": 3}
KNOWN_CURRENCIES = {"AUD", "CAD", "CHF", "CNY", "EUR", "GBP", "JPY", "NZD", "USD"}
SYMBOL_CURRENCY_FALLBACKS = {
    "GOLD": ["USD"],
    "XAUUSD": ["USD"],
    "US30": ["USD"],
    "US100": ["USD"],
    "BTCUSD": ["USD"],
    "ETHUSD": ["USD"],
}


@dataclass(frozen=True, slots=True)
class EventRuleProfile:
    tag: str
    family: str
    priority: int
    any_keywords: tuple[str, ...] = ()
    all_keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    pre_news_block_minutes: int | None = None
    pre_news_close_only_minutes: int | None = None
    release_window_before_minutes: int | None = None
    release_window_after_minutes: int | None = None
    post_release_freeze_minutes: int | None = None
    post_news_cooldown_minutes: int | None = None
    post_news_reentry_minutes: int | None = None
    strict_spread_limit_points: int | None = None


EVENT_RULE_PROFILES: tuple[EventRuleProfile, ...] = (
    EventRuleProfile(
        tag="POWELL_PRESSER",
        family="FOMC",
        priority=60,
        any_keywords=("POWELL", "FOMC", "FEDERAL RESERVE", "FED"),
        all_keywords=("PRESS CONFERENCE",),
        pre_news_block_minutes=120,
        pre_news_close_only_minutes=20,
        post_release_freeze_minutes=15,
        post_news_cooldown_minutes=45,
        post_news_reentry_minutes=90,
        strict_spread_limit_points=34,
    ),
    EventRuleProfile(
        tag="FOMC_RATE_DECISION",
        family="FOMC",
        priority=56,
        any_keywords=(
            "FEDERAL FUNDS RATE",
            "INTEREST RATE DECISION",
            "RATE DECISION",
            "POLICY RATE",
        ),
        pre_news_block_minutes=105,
        pre_news_close_only_minutes=18,
        post_release_freeze_minutes=12,
        post_news_cooldown_minutes=35,
        post_news_reentry_minutes=70,
        strict_spread_limit_points=36,
    ),
    EventRuleProfile(
        tag="FOMC_STATEMENT",
        family="FOMC",
        priority=53,
        any_keywords=(
            "FOMC STATEMENT",
            "RATE STATEMENT",
            "MONETARY POLICY STATEMENT",
            "STATEMENT",
        ),
        all_keywords=("FOMC",),
        pre_news_block_minutes=90,
        pre_news_close_only_minutes=15,
        post_release_freeze_minutes=10,
        post_news_cooldown_minutes=30,
        post_news_reentry_minutes=60,
        strict_spread_limit_points=38,
    ),
    EventRuleProfile(
        tag="FOMC_MINUTES",
        family="FOMC",
        priority=49,
        any_keywords=(
            "FOMC MINUTES",
            "MEETING MINUTES",
            "FEDERAL OPEN MARKET COMMITTEE MINUTES",
        ),
        all_keywords=("MINUTES",),
        pre_news_block_minutes=60,
        pre_news_close_only_minutes=10,
        post_release_freeze_minutes=6,
        post_news_cooldown_minutes=20,
        post_news_reentry_minutes=40,
        strict_spread_limit_points=39,
    ),
    EventRuleProfile(
        tag="FOMC",
        family="FOMC",
        priority=50,
        any_keywords=("FOMC", "FEDERAL RESERVE", "FED"),
        exclude_keywords=("PRESS CONFERENCE", "STATEMENT", "RATE DECISION", "MINUTES"),
        pre_news_block_minutes=90,
        pre_news_close_only_minutes=15,
        post_release_freeze_minutes=10,
        post_news_cooldown_minutes=30,
        post_news_reentry_minutes=60,
        strict_spread_limit_points=38,
    ),
    EventRuleProfile(
        tag="NFP_PAYROLLS",
        family="NFP",
        priority=55,
        any_keywords=(
            "NON-FARM PAYROLLS",
            "NONFARM PAYROLLS",
            "NON-FARM EMPLOYMENT CHANGE",
            "PAYROLLS",
        ),
        exclude_keywords=("ADP",),
        pre_news_block_minutes=70,
        pre_news_close_only_minutes=12,
        post_release_freeze_minutes=6,
        post_news_cooldown_minutes=30,
        post_news_reentry_minutes=50,
        strict_spread_limit_points=38,
    ),
    EventRuleProfile(
        tag="NFP_WAGES",
        family="NFP",
        priority=52,
        any_keywords=("AVERAGE HOURLY EARNINGS", "AVERAGE WEEKLY EARNINGS"),
        pre_news_block_minutes=60,
        pre_news_close_only_minutes=10,
        post_release_freeze_minutes=5,
        post_news_cooldown_minutes=25,
        post_news_reentry_minutes=45,
        strict_spread_limit_points=39,
    ),
    EventRuleProfile(
        tag="NFP_UNEMPLOYMENT",
        family="NFP",
        priority=51,
        any_keywords=("UNEMPLOYMENT RATE", "JOBLESS RATE"),
        pre_news_block_minutes=60,
        pre_news_close_only_minutes=10,
        post_release_freeze_minutes=5,
        post_news_cooldown_minutes=25,
        post_news_reentry_minutes=45,
        strict_spread_limit_points=40,
    ),
    EventRuleProfile(
        tag="CPI_CORE",
        family="CPI",
        priority=48,
        any_keywords=(
            "CORE CPI",
            "CORE CONSUMER PRICE",
            "CORE INFLATION",
        ),
        exclude_keywords=("PCE",),
        pre_news_block_minutes=50,
        pre_news_close_only_minutes=10,
        post_release_freeze_minutes=5,
        post_news_cooldown_minutes=25,
        post_news_reentry_minutes=40,
        strict_spread_limit_points=40,
    ),
    EventRuleProfile(
        tag="CORE_PCE",
        family="CPI",
        priority=49,
        any_keywords=("CORE PCE", "CORE PCE PRICE INDEX", "CORE PERSONAL CONSUMPTION EXPENDITURES"),
        pre_news_block_minutes=55,
        pre_news_close_only_minutes=10,
        post_release_freeze_minutes=5,
        post_news_cooldown_minutes=25,
        post_news_reentry_minutes=45,
        strict_spread_limit_points=39,
    ),
    EventRuleProfile(
        tag="CPI_MM",
        family="CPI",
        priority=47,
        any_keywords=("CPI M/M", "CONSUMER PRICE INDEX M/M", "INFLATION M/M"),
        pre_news_block_minutes=48,
        pre_news_close_only_minutes=9,
        post_release_freeze_minutes=5,
        post_news_cooldown_minutes=22,
        post_news_reentry_minutes=38,
        strict_spread_limit_points=40,
    ),
    EventRuleProfile(
        tag="CPI_YY",
        family="CPI",
        priority=46,
        any_keywords=("CPI Y/Y", "CONSUMER PRICE INDEX Y/Y", "INFLATION Y/Y"),
        pre_news_block_minutes=45,
        pre_news_close_only_minutes=8,
        post_release_freeze_minutes=4,
        post_news_cooldown_minutes=20,
        post_news_reentry_minutes=35,
        strict_spread_limit_points=41,
    ),
    EventRuleProfile(
        tag="CPI_HEADLINE",
        family="CPI",
        priority=45,
        any_keywords=(
            "CONSUMER PRICE INDEX",
            "INFLATION RATE",
        ),
        exclude_keywords=("Y/Y", "M/M", "CORE", "PCE"),
        pre_news_block_minutes=45,
        pre_news_close_only_minutes=8,
        post_release_freeze_minutes=4,
        post_news_cooldown_minutes=20,
        post_news_reentry_minutes=35,
        strict_spread_limit_points=41,
    ),
    EventRuleProfile(
        tag="CPI",
        family="CPI",
        priority=44,
        any_keywords=("CPI", "CONSUMER PRICE", "PCE PRICE", "INFLATION"),
        exclude_keywords=("Y/Y", "M/M", "CORE", "PCE"),
        pre_news_block_minutes=45,
        pre_news_close_only_minutes=8,
        post_release_freeze_minutes=4,
        post_news_cooldown_minutes=20,
        post_news_reentry_minutes=35,
        strict_spread_limit_points=42,
    ),
    EventRuleProfile(
        tag="PPI_CORE",
        family="PPI",
        priority=41,
        any_keywords=("CORE PPI", "CORE PRODUCER PRICE", "CORE PRODUCER PRICE INDEX"),
        pre_news_block_minutes=35,
        pre_news_close_only_minutes=7,
        post_release_freeze_minutes=4,
        post_news_cooldown_minutes=15,
        post_news_reentry_minutes=25,
        strict_spread_limit_points=48,
    ),
    EventRuleProfile(
        tag="PPI",
        family="PPI",
        priority=40,
        any_keywords=("PPI", "PRODUCER PRICE INDEX", "PRODUCER PRICES"),
        exclude_keywords=("CORE",),
        pre_news_block_minutes=30,
        pre_news_close_only_minutes=6,
        post_release_freeze_minutes=4,
        post_news_cooldown_minutes=15,
        post_news_reentry_minutes=25,
        strict_spread_limit_points=50,
    ),
    EventRuleProfile(
        tag="RETAIL_SALES_CONTROL",
        family="RETAIL_SALES",
        priority=39,
        any_keywords=("RETAIL SALES CONTROL GROUP", "CORE RETAIL SALES", "CONTROL GROUP"),
        pre_news_block_minutes=30,
        pre_news_close_only_minutes=6,
        post_release_freeze_minutes=4,
        post_news_cooldown_minutes=12,
        post_news_reentry_minutes=20,
        strict_spread_limit_points=52,
    ),
    EventRuleProfile(
        tag="RETAIL_SALES",
        family="RETAIL_SALES",
        priority=38,
        any_keywords=("RETAIL SALES",),
        exclude_keywords=("CONTROL GROUP", "CORE"),
        pre_news_block_minutes=25,
        pre_news_close_only_minutes=5,
        post_release_freeze_minutes=3,
        post_news_cooldown_minutes=10,
        post_news_reentry_minutes=18,
        strict_spread_limit_points=54,
    ),
    EventRuleProfile(
        tag="ISM_SERVICES",
        family="ISM",
        priority=37,
        any_keywords=("ISM SERVICES", "NON-MANUFACTURING PMI", "SERVICES PMI"),
        pre_news_block_minutes=25,
        pre_news_close_only_minutes=5,
        post_release_freeze_minutes=3,
        post_news_cooldown_minutes=10,
        post_news_reentry_minutes=15,
        strict_spread_limit_points=58,
    ),
    EventRuleProfile(
        tag="ISM_MANUFACTURING",
        family="ISM",
        priority=36,
        any_keywords=("ISM MANUFACTURING", "MANUFACTURING PMI"),
        exclude_keywords=("NON-MANUFACTURING", "SERVICES"),
        pre_news_block_minutes=25,
        pre_news_close_only_minutes=5,
        post_release_freeze_minutes=3,
        post_news_cooldown_minutes=10,
        post_news_reentry_minutes=15,
        strict_spread_limit_points=60,
    ),
    EventRuleProfile(
        tag="ADP_EMPLOYMENT",
        family="LABOR",
        priority=35,
        any_keywords=("ADP EMPLOYMENT", "ADP NON-FARM EMPLOYMENT", "ADP PAYROLLS"),
        pre_news_block_minutes=25,
        pre_news_close_only_minutes=5,
        post_release_freeze_minutes=3,
        post_news_cooldown_minutes=10,
        post_news_reentry_minutes=15,
        strict_spread_limit_points=58,
    ),
    EventRuleProfile(
        tag="JOLTS_JOB_OPENINGS",
        family="LABOR",
        priority=34,
        any_keywords=("JOLTS JOB OPENINGS", "JOB OPENINGS", "JOLTS"),
        pre_news_block_minutes=20,
        pre_news_close_only_minutes=4,
        post_release_freeze_minutes=3,
        post_news_cooldown_minutes=8,
        post_news_reentry_minutes=15,
        strict_spread_limit_points=60,
    ),
    EventRuleProfile(
        tag="INITIAL_JOBLESS_CLAIMS",
        family="LABOR",
        priority=33,
        any_keywords=("INITIAL JOBLESS CLAIMS", "UNEMPLOYMENT CLAIMS", "JOBLESS CLAIMS"),
        pre_news_block_minutes=20,
        pre_news_close_only_minutes=4,
        post_release_freeze_minutes=3,
        post_news_cooldown_minutes=8,
        post_news_reentry_minutes=15,
        strict_spread_limit_points=62,
    ),
    EventRuleProfile(
        tag="CONSUMER_SENTIMENT",
        family="SENTIMENT",
        priority=32,
        any_keywords=(
            "CONSUMER SENTIMENT",
            "MICHIGAN CONSUMER SENTIMENT",
            "CONSUMER CONFIDENCE",
        ),
        pre_news_block_minutes=20,
        pre_news_close_only_minutes=4,
        post_release_freeze_minutes=3,
        post_news_cooldown_minutes=8,
        post_news_reentry_minutes=12,
        strict_spread_limit_points=64,
    ),
)


@dataclass(slots=True)
class NewsEvent:
    title: str
    currency: str
    timestamp: datetime
    impact: str
    importance: int
    forecast: str = ""
    previous: str = ""

    def minutes_to_event(self, now: datetime) -> float:
        return (self.timestamp - now).total_seconds() / 60.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat(),
            "impact": self.impact,
            "importance": self.importance,
            "forecast": self.forecast,
            "previous": self.previous,
        }


@dataclass(slots=True)
class NewsGuardDecision:
    symbol: str
    blocked: bool
    close_positions: bool
    reason: str
    phase: str = "none"
    event_tag: str = "GENERIC"
    event_family: str = "GENERIC"
    event_priority: int = 0
    strict_spread_limit_points: int | None = None
    event: NewsEvent | None = None
    long_blocked: bool = False
    sentiment_score: int | None = None
    sentiment_label: str | None = None

    def payload(self) -> dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "blocked": self.blocked,
            "close_positions": self.close_positions,
            "reason": self.reason,
            "phase": self.phase,
            "event_tag": self.event_tag,
            "event_family": self.event_family,
            "event_priority": self.event_priority,
            "strict_spread_limit_points": self.strict_spread_limit_points,
            "long_blocked": self.long_blocked,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
        }
        if self.event is not None:
            payload["event"] = self.event.to_payload()
        return payload


class NewsFilter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cached_events: list[NewsEvent] = []
        self._cache_expires_at: datetime | None = None
        self._crypto_guard_until_by_symbol: dict[str, datetime] = {}
        self._calendar_backoff_until: datetime | None = None
        self._calendar_backoff_seconds: int = 0
        self._calendar_last_warning_key: str | None = None
        self._calendar_last_warning_message: str | None = None
        self._calendar_last_warning_at: datetime | None = None
        self.settings.news_cache_path.parent.mkdir(parents=True, exist_ok=True)

    def internet_available(self) -> bool:
        if not self.settings.news.enabled:
            return True
        now = datetime.now(timezone.utc)
        if self._calendar_backoff_active(now):
            return True
        if self._cached_events:
            return True
        if self._load_disk_cache(now):
            return True
        try:
            response = requests.get(
                self.settings.news.calendar_url,
                timeout=max(2.0, min(10.0, self.settings.news.timeout_seconds)),
            )
            return 200 <= int(response.status_code) < 500
        except Exception:
            return False

    def _is_crypto_symbol(self, symbol: str) -> bool:
        return symbol.upper() in {item.upper() for item in self.settings.trading.crypto_symbols}

    def _extract_sentiment_score(self, payload: dict[str, Any]) -> int | None:
        for key in ("sentiment_score", "score", "vote"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                return int(float(value))
            except Exception:
                continue
        votes = payload.get("votes")
        if isinstance(votes, dict):
            positive = int(votes.get("positive") or 0)
            negative = int(votes.get("negative") or 0)
            total = positive + negative
            if total > 0:
                return int(round((positive / total) * 100))
        if isinstance(votes, (int, float)):
            try:
                return int(float(votes))
            except Exception:
                return None
        return None

    def _extract_sentiment_label(self, payload: dict[str, Any]) -> str | None:
        for key in ("sentiment", "sentiment_label", "label"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
            if isinstance(value, dict):
                title = value.get("title")
                if isinstance(title, str) and title.strip():
                    return title.strip().lower()
        votes = payload.get("votes")
        if isinstance(votes, dict):
            positive = int(votes.get("positive") or 0)
            negative = int(votes.get("negative") or 0)
            if negative >= max(3, positive * 2):
                return "very_bearish"
            if negative > positive:
                return "bearish"
            if positive >= max(3, negative * 2):
                return "bullish"
            if positive > negative:
                return "bullish"
        return None

    def _post_text(self, payload: dict[str, Any]) -> str:
        text_parts = [
            str(payload.get("title") or ""),
            str(payload.get("description") or ""),
        ]
        content = payload.get("content")
        if isinstance(content, dict):
            text_parts.append(str(content.get("clean") or ""))
            text_parts.append(str(content.get("original") or ""))
        return " ".join(part for part in text_parts if part).strip()

    def _post_timestamp(self, payload: dict[str, Any]) -> datetime | None:
        for key in ("published_at", "created_at"):
            raw_value = payload.get(key)
            if not raw_value:
                continue
            try:
                normalized = str(raw_value).replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(normalized)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return timestamp.astimezone(timezone.utc)
            except Exception:
                continue
        return None

    def _post_negative_votes(self, payload: dict[str, Any]) -> int:
        votes = payload.get("votes")
        if not isinstance(votes, dict):
            return 0
        total = 0
        for key in ("negative", "important", "toxic"):
            try:
                total += int(votes.get(key) or 0)
            except Exception:
                continue
        return total

    def _post_panic_score(self, payload: dict[str, Any]) -> int | None:
        for key in ("panic_score", "panic_score_1h"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                return int(float(value))
            except Exception:
                continue
        return None

    def _post_instrument_codes(self, payload: dict[str, Any]) -> set[str]:
        codes: set[str] = set()
        instruments = payload.get("instruments")
        if isinstance(instruments, list):
            for item in instruments:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code") or "").strip().upper()
                if code:
                    codes.add(code)
        return codes

    def _symbol_instrument_codes(self, symbol: str) -> set[str]:
        normalized = symbol.upper()
        mapping = {
            "BTCUSD": {"BTC"},
            "ETHUSD": {"ETH"},
        }
        return mapping.get(normalized, {normalized[:3]} if len(normalized) >= 3 else set())

    def _post_matches_symbol(self, symbol: str, payload: dict[str, Any]) -> bool:
        instruments = self._post_instrument_codes(payload)
        symbol_codes = self._symbol_instrument_codes(symbol)
        if instruments:
            return bool(instruments & symbol_codes)

        text = self._post_text(payload).upper()
        aliases = {
            "BTCUSD": ("BTC", "BITCOIN"),
            "ETHUSD": ("ETH", "ETHEREUM"),
        }
        return any(alias in text for alias in aliases.get(symbol.upper(), (symbol.upper(),)))

    def _post_is_recent(self, payload: dict[str, Any], current_time: datetime) -> bool:
        timestamp = self._post_timestamp(payload)
        if timestamp is None:
            return True
        age_minutes = (current_time - timestamp).total_seconds() / 60.0
        return age_minutes <= max(1, self.settings.news.crypto_keyword_guard_lookback_minutes)

    def _cryptopanic_currency_codes(self, symbol: str) -> list[str]:
        normalized = symbol.upper()
        codes: list[str] = []
        if len(normalized) >= 3:
            base_code = normalized[:3]
            if base_code.isalpha():
                codes.append(base_code)
        if normalized == "ETHUSD" and "BTC" not in codes:
            codes.append("BTC")
        return codes or ["BTC", "ETH"]

    def _raise_cryptopanic_error(self, response: requests.Response) -> None:
        status_code = int(response.status_code)
        status_messages = {
            401: "CryptoPanic rejected auth_token. Verify NEWS__CRYPTOPANIC_API_KEY.",
            403: "CryptoPanic denied access or rate-limited this endpoint for the current API plan.",
            404: "CryptoPanic endpoint was not found. Verify NEWS__CRYPTOPANIC_API_URL matches the current API version.",
            429: "CryptoPanic rate limit exceeded. Reduce polling frequency or wait for the limit window to reset.",
            500: "CryptoPanic returned an internal server error.",
        }
        message = status_messages.get(
            status_code,
            f"CryptoPanic request failed with HTTP {status_code}.",
        )
        raise RuntimeError(message)

    def _cryptopanic_payload(self, symbol: str) -> list[dict[str, Any]]:
        if not self.settings.news.cryptopanic_enabled or not self.settings.news.cryptopanic_api_key:
            return []
        params: dict[str, Any] = {
            "auth_token": self.settings.news.cryptopanic_api_key,
            "currencies": ",".join(self._cryptopanic_currency_codes(symbol)),
        }
        if self.settings.news.cryptopanic_public:
            params["public"] = "true"
        if self.settings.news.cryptopanic_regions.strip():
            params["regions"] = self.settings.news.cryptopanic_regions.strip()
        if self.settings.news.cryptopanic_filter.strip():
            params["filter"] = self.settings.news.cryptopanic_filter.strip()
        if self.settings.news.cryptopanic_kind.strip():
            params["kind"] = self.settings.news.cryptopanic_kind.strip()
        response = requests.get(
            self.settings.news.cryptopanic_api_url,
            params=params,
            timeout=max(2.0, min(10.0, self.settings.news.timeout_seconds)),
        )
        if int(response.status_code) >= 400:
            self._raise_cryptopanic_error(response)
        payload = response.json()
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
        return []

    def _crypto_sentiment_guard(self, symbol: str, current_time: datetime) -> NewsGuardDecision | None:
        if not self._is_crypto_symbol(symbol):
            return None
        until = self._crypto_guard_until_by_symbol.get(symbol.upper())
        if until is not None and current_time < until:
            return NewsGuardDecision(
                symbol=symbol,
                blocked=True,
                close_positions=False,
                reason=f"crypto_keyword_guard_active_until:{until.isoformat()}",
                phase="crypto_keyword_guard",
                event_tag="CRYPTO_KEYWORD",
                event_family="CRYPTO_SENTIMENT",
                long_blocked=True,
            )

        try:
            posts = self._cryptopanic_payload(symbol)
        except Exception as exc:
            logger.warning("CryptoPanic fetch failed for {}: {}", symbol, exc)
            return None
        if not posts:
            return None

        keyword_terms = {term.upper() for term in self.settings.news.crypto_keyword_guard_terms}
        excluded_terms = {term.upper() for term in self.settings.news.crypto_keyword_guard_excluded_terms}
        best_score: int | None = None
        best_label: str | None = None
        for post in posts:
            if not self._post_is_recent(post, current_time):
                continue
            if (
                self.settings.news.crypto_keyword_guard_require_instrument_match
                and not self._post_matches_symbol(symbol, post)
            ):
                continue
            combined_text = self._post_text(post).upper()
            if any(term in combined_text for term in excluded_terms):
                continue
            panic_score = self._post_panic_score(post)
            negative_votes = self._post_negative_votes(post)
            label = self._extract_sentiment_label(post)
            severe_keyword_match = any(term in combined_text for term in keyword_terms)
            severe_context = (
                (panic_score is not None and panic_score >= self.settings.news.crypto_keyword_guard_min_panic_score)
                or negative_votes >= self.settings.news.crypto_keyword_guard_min_negative_votes
                or label in {"bearish", "very_bearish"}
            )
            if severe_keyword_match and severe_context:
                until_time = current_time + timedelta(minutes=self.settings.news.crypto_bearish_block_minutes)
                self._crypto_guard_until_by_symbol[symbol.upper()] = until_time
                return NewsGuardDecision(
                    symbol=symbol,
                    blocked=True,
                    close_positions=True,
                    reason=f"crypto_keyword_guard:{str(post.get('title') or combined_text[:100]).upper()[:120]}",
                    phase="crypto_keyword_guard",
                    event_tag="CRYPTO_KEYWORD",
                    event_family="CRYPTO_SENTIMENT",
                    long_blocked=True,
                    sentiment_score=panic_score,
                    sentiment_label=label,
                )

            score = self._extract_sentiment_score(post)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_label = label
            elif best_label is None and label is not None:
                best_label = label

        very_bearish = (best_label or "") == self.settings.news.cryptopanic_very_bearish_label.lower()
        score_bearish = best_score is not None and best_score < self.settings.news.cryptopanic_min_sentiment_score
        if very_bearish or score_bearish:
            return NewsGuardDecision(
                symbol=symbol,
                blocked=True,
                close_positions=True,
                reason=(
                    f"crypto_sentiment_guard:label={best_label or 'unknown'}:"
                    f"score={best_score if best_score is not None else 'na'}"
                ),
                phase="crypto_sentiment_guard",
                event_tag="CRYPTO_SENTIMENT",
                event_family="CRYPTO_SENTIMENT",
                long_blocked=True,
                sentiment_score=best_score,
                sentiment_label=best_label,
            )
        return None

    def _default_rule_profile(self) -> EventRuleProfile:
        return EventRuleProfile(
            tag="GENERIC",
            family="GENERIC",
            priority=10,
            pre_news_block_minutes=self.settings.news.pre_news_block_minutes,
            pre_news_close_only_minutes=self.settings.news.pre_news_close_only_minutes,
            release_window_before_minutes=self.settings.news.release_window_before_minutes,
            release_window_after_minutes=self.settings.news.release_window_after_minutes,
            post_release_freeze_minutes=self.settings.news.post_release_freeze_minutes,
            post_news_cooldown_minutes=self.settings.news.post_news_cooldown_minutes,
            post_news_reentry_minutes=self.settings.news.post_news_reentry_minutes,
            strict_spread_limit_points=self.settings.risk.news_spread_limit_points,
        )

    def _matches_profile(self, title: str, profile: EventRuleProfile) -> bool:
        if profile.exclude_keywords and any(keyword in title for keyword in profile.exclude_keywords):
            return False
        if profile.all_keywords and not all(keyword in title for keyword in profile.all_keywords):
            return False
        if profile.any_keywords and not any(keyword in title for keyword in profile.any_keywords):
            return False
        return bool(profile.any_keywords or profile.all_keywords)

    def _rule_profile_for_event(self, event: NewsEvent) -> EventRuleProfile:
        title = event.title.upper()
        for profile in EVENT_RULE_PROFILES:
            if self._matches_profile(title, profile):
                return EventRuleProfile(
                    tag=profile.tag,
                    family=profile.family,
                    priority=profile.priority,
                    any_keywords=profile.any_keywords,
                    all_keywords=profile.all_keywords,
                    exclude_keywords=profile.exclude_keywords,
                    pre_news_block_minutes=profile.pre_news_block_minutes
                    or self.settings.news.pre_news_block_minutes,
                    pre_news_close_only_minutes=profile.pre_news_close_only_minutes
                    or self.settings.news.pre_news_close_only_minutes,
                    release_window_before_minutes=profile.release_window_before_minutes
                    or self.settings.news.release_window_before_minutes,
                    release_window_after_minutes=profile.release_window_after_minutes
                    or self.settings.news.release_window_after_minutes,
                    post_release_freeze_minutes=profile.post_release_freeze_minutes
                    or self.settings.news.post_release_freeze_minutes,
                    post_news_cooldown_minutes=profile.post_news_cooldown_minutes
                    or self.settings.news.post_news_cooldown_minutes,
                    post_news_reentry_minutes=profile.post_news_reentry_minutes
                    or self.settings.news.post_news_reentry_minutes,
                    strict_spread_limit_points=profile.strict_spread_limit_points
                    or self.settings.risk.news_spread_limit_points,
                )
        return self._default_rule_profile()

    def _decision_for_event(
        self,
        symbol: str,
        event: NewsEvent,
        current_time: datetime,
    ) -> tuple[int, int, float, NewsGuardDecision]:
        minutes_to_event = event.minutes_to_event(current_time)
        rule = self._rule_profile_for_event(event)
        strict_limit = rule.strict_spread_limit_points or self.settings.risk.news_spread_limit_points or None

        if (
            -(rule.release_window_after_minutes or 0)
            <= minutes_to_event
            <= (rule.release_window_before_minutes or 0)
        ):
            return (
                6,
                rule.priority,
                abs(minutes_to_event),
                NewsGuardDecision(
                    symbol=symbol,
                    blocked=True,
                    close_positions=True,
                    reason=(
                        f"release_minute:{rule.tag}:{event.currency}:{event.title}:{event.impact}:"
                        f"{minutes_to_event:+.0f}m"
                    ),
                    phase="release_minute",
                    event_tag=rule.tag,
                    event_family=rule.family,
                    event_priority=rule.priority,
                    strict_spread_limit_points=strict_limit,
                    event=event,
                ),
            )

        if 0 <= minutes_to_event <= (rule.pre_news_close_only_minutes or 0):
            return (
                5,
                rule.priority,
                abs(minutes_to_event),
                NewsGuardDecision(
                    symbol=symbol,
                    blocked=True,
                    close_positions=True,
                    reason=(
                        f"pre_news_close_only:{rule.tag}:{event.currency}:{event.title}:{event.impact}:"
                        f"{minutes_to_event:+.0f}m"
                    ),
                    phase="pre_news_close_only",
                    event_tag=rule.tag,
                    event_family=rule.family,
                    event_priority=rule.priority,
                    strict_spread_limit_points=strict_limit,
                    event=event,
                ),
            )

        if 0 <= minutes_to_event <= (rule.pre_news_block_minutes or 0):
            return (
                4,
                rule.priority,
                abs(minutes_to_event),
                NewsGuardDecision(
                    symbol=symbol,
                    blocked=True,
                    close_positions=False,
                    reason=(
                        f"pre_news:{rule.tag}:{event.currency}:{event.title}:{event.impact}:"
                        f"{minutes_to_event:+.0f}m"
                    ),
                    phase="pre_news",
                    event_tag=rule.tag,
                    event_family=rule.family,
                    event_priority=rule.priority,
                    strict_spread_limit_points=strict_limit,
                    event=event,
                ),
            )

        if -((rule.post_release_freeze_minutes or 0) + (rule.release_window_after_minutes or 0)) <= minutes_to_event < -(
            rule.release_window_after_minutes or 0
        ):
            return (
                3,
                rule.priority,
                abs(minutes_to_event),
                NewsGuardDecision(
                    symbol=symbol,
                    blocked=True,
                    close_positions=False,
                    reason=(
                        f"post_release_freeze:{rule.tag}:{event.currency}:{event.title}:{event.impact}:"
                        f"{minutes_to_event:+.0f}m"
                    ),
                    phase="post_release_freeze",
                    event_tag=rule.tag,
                    event_family=rule.family,
                    event_priority=rule.priority,
                    strict_spread_limit_points=strict_limit,
                    event=event,
                ),
            )

        if -((rule.post_news_cooldown_minutes or 0)) <= minutes_to_event < -(
            (rule.post_release_freeze_minutes or 0) + (rule.release_window_after_minutes or 0)
        ):
            return (
                2,
                rule.priority,
                abs(minutes_to_event),
                NewsGuardDecision(
                    symbol=symbol,
                    blocked=True,
                    close_positions=False,
                    reason=(
                        f"post_news_cooldown:{rule.tag}:{event.currency}:{event.title}:{event.impact}:"
                        f"{minutes_to_event:+.0f}m"
                    ),
                    phase="post_news_cooldown",
                    event_tag=rule.tag,
                    event_family=rule.family,
                    event_priority=rule.priority,
                    strict_spread_limit_points=strict_limit,
                    event=event,
                ),
            )

        if -((rule.post_news_cooldown_minutes or 0) + (rule.post_news_reentry_minutes or 0)) <= minutes_to_event < -(
            rule.post_news_cooldown_minutes or 0
        ):
            return (
                1,
                rule.priority,
                abs(minutes_to_event),
                NewsGuardDecision(
                    symbol=symbol,
                    blocked=False,
                    close_positions=False,
                    reason=(
                        f"post_news_reentry:{rule.tag}:{event.currency}:{event.title}:{event.impact}:"
                        f"{minutes_to_event:+.0f}m"
                    ),
                    phase="post_news_reentry",
                    event_tag=rule.tag,
                    event_family=rule.family,
                    event_priority=rule.priority,
                    strict_spread_limit_points=strict_limit,
                    event=event,
                ),
            )

        return (
            0,
            rule.priority,
            abs(minutes_to_event),
            NewsGuardDecision(
                symbol=symbol,
                blocked=False,
                close_positions=False,
                reason=(
                    f"news_watch:{rule.tag}:{event.currency}:{event.title}:{event.impact}:{minutes_to_event:+.0f}m"
                ),
                phase="watch",
                event_tag=rule.tag,
                event_family=rule.family,
                event_priority=rule.priority,
                strict_spread_limit_points=strict_limit,
                event=event,
            ),
        )

    def _serialize_events(self, events: list[NewsEvent], fetched_at: datetime) -> dict[str, Any]:
        return {
            "fetched_at": fetched_at.astimezone(timezone.utc).isoformat(),
            "events": [event.to_payload() for event in events],
        }

    def _write_disk_cache(self, events: list[NewsEvent], fetched_at: datetime) -> None:
        payload = self._serialize_events(events, fetched_at)
        self.settings.news_cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _load_disk_cache(self, now: datetime) -> list[NewsEvent]:
        if not self.settings.news_cache_path.exists():
            return []

        try:
            payload = json.loads(self.settings.news_cache_path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            age_seconds = (now - fetched_at.astimezone(timezone.utc)).total_seconds()
            if age_seconds > self.settings.news.stale_cache_max_seconds:
                return []

            events: list[NewsEvent] = []
            for item in payload.get("events", []):
                timestamp = datetime.fromisoformat(item["timestamp"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        title=str(item.get("title") or "").strip(),
                        currency=str(item.get("currency") or "").strip().upper(),
                        timestamp=timestamp.astimezone(timezone.utc),
                        impact=str(item.get("impact") or "").strip(),
                        importance=int(item.get("importance") or 0),
                        forecast=str(item.get("forecast") or "").strip(),
                        previous=str(item.get("previous") or "").strip(),
                    )
                )
            return sorted(events, key=lambda item: item.timestamp)
        except Exception as exc:
            logger.warning("Failed to read news disk cache: {}", exc)
            return []

    def _fetch_events(self) -> list[NewsEvent]:
        response = httpx.get(
            self.settings.news.calendar_url,
            timeout=self.settings.news.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "MT5AI/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        events: list[NewsEvent] = []
        for item in payload:
            try:
                raw_timestamp = item.get("date")
                if not raw_timestamp:
                    continue
                timestamp = datetime.fromisoformat(raw_timestamp)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                timestamp = timestamp.astimezone(timezone.utc)

                impact = str(item.get("impact") or "").strip().lower()
                importance = IMPACT_SCORES.get(impact, 0)
                currency = str(item.get("country") or "").strip().upper()
                title = str(item.get("title") or "").strip()
                if not currency or not title or importance <= 0:
                    continue

                events.append(
                    NewsEvent(
                        title=title,
                        currency=currency,
                        timestamp=timestamp,
                        impact=impact.title(),
                        importance=importance,
                        forecast=str(item.get("forecast") or "").strip(),
                        previous=str(item.get("previous") or "").strip(),
                    )
                )
            except Exception as exc:
                logger.debug("Skipping malformed news payload item: {}", exc)
        return sorted(events, key=lambda item: item.timestamp)

    def _calendar_error_status_code(self, exc: Exception) -> int | None:
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                return int(exc.response.status_code)
            except Exception:
                return None
        text = str(exc)
        for code in ("429", "500", "502", "503", "504"):
            if code in text:
                return int(code)
        return None

    def _calendar_retry_after_seconds(self, exc: Exception, now: datetime) -> int | None:
        if not isinstance(exc, httpx.HTTPStatusError):
            return None
        header_value = str(exc.response.headers.get("Retry-After") or "").strip()
        if not header_value:
            return None
        try:
            seconds = int(float(header_value))
            return max(0, seconds)
        except Exception:
            pass
        try:
            retry_at = parsedate_to_datetime(header_value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0, int((retry_at.astimezone(timezone.utc) - now).total_seconds()))
        except Exception:
            return None

    def _calendar_backoff_active(self, now: datetime) -> bool:
        return self._calendar_backoff_until is not None and now < self._calendar_backoff_until

    def _calendar_backoff_message(self, now: datetime) -> str:
        if self._calendar_backoff_until is None:
            return "Calendar backoff active"
        remaining_seconds = max(
            0,
            int((self._calendar_backoff_until - now).total_seconds()),
        )
        return f"Calendar feed backoff active; using cached events for another {remaining_seconds}s"

    def _log_calendar_warning_once(self, now: datetime, message: str, key: str | None = None) -> None:
        interval = max(0, int(self.settings.news.calendar_warning_interval_seconds))
        dedupe_key = key or message
        if (
            self._calendar_last_warning_key == dedupe_key
            and self._calendar_last_warning_at is not None
            and (now - self._calendar_last_warning_at).total_seconds() < interval
        ):
            return
        logger.warning(message)
        self._calendar_last_warning_key = dedupe_key
        self._calendar_last_warning_message = message
        self._calendar_last_warning_at = now

    def _reset_calendar_backoff(self) -> None:
        self._calendar_backoff_until = None
        self._calendar_backoff_seconds = 0
        self._calendar_last_warning_key = None
        self._calendar_last_warning_message = None
        self._calendar_last_warning_at = None

    def _apply_calendar_backoff(self, now: datetime, exc: Exception) -> None:
        status_code = self._calendar_error_status_code(exc)
        retryable = status_code in {429, 500, 502, 503, 504}
        if not retryable:
            return
        base = max(1, int(self.settings.news.calendar_backoff_base_seconds))
        maximum = max(base, int(self.settings.news.calendar_backoff_max_seconds))
        retry_after_seconds = self._calendar_retry_after_seconds(exc, now)
        if retry_after_seconds is not None:
            self._calendar_backoff_seconds = min(maximum, max(base, retry_after_seconds))
        elif self._calendar_backoff_seconds <= 0:
            self._calendar_backoff_seconds = base
        else:
            self._calendar_backoff_seconds = min(maximum, self._calendar_backoff_seconds * 2)
        self._calendar_backoff_until = now + timedelta(seconds=self._calendar_backoff_seconds)

    def _events(self, now: datetime) -> list[NewsEvent]:
        if self._cache_expires_at is not None and now < self._cache_expires_at and self._cached_events:
            return self._cached_events

        disk_cached = self._load_disk_cache(now)
        if self._calendar_backoff_active(now) and (self._cached_events or disk_cached):
            cached = self._cached_events or disk_cached
            if disk_cached and not self._cached_events:
                self._cached_events = disk_cached
            self._cache_expires_at = max(
                now + timedelta(seconds=self.settings.news.cache_ttl_seconds),
                self._calendar_backoff_until or now,
            )
            self._log_calendar_warning_once(now, self._calendar_backoff_message(now), key="calendar_backoff_active")
            return cached

        try:
            events = self._fetch_events()
            self._reset_calendar_backoff()
            self._cached_events = events
            self._cache_expires_at = now + timedelta(seconds=self.settings.news.cache_ttl_seconds)
            self._write_disk_cache(events, now)
            return events
        except Exception as exc:
            self._apply_calendar_backoff(now, exc)
            if self._cached_events:
                self._cache_expires_at = max(
                    now + timedelta(seconds=self.settings.news.cache_ttl_seconds),
                    self._calendar_backoff_until or now,
                )
                status_code = self._calendar_error_status_code(exc)
                error_key = f"calendar_fetch_cached_{status_code or 'unknown'}"
                self._log_calendar_warning_once(
                    now,
                    f"News fetch failed, using cached events: {exc}",
                    key=error_key,
                )
                return self._cached_events
            if disk_cached:
                status_code = self._calendar_error_status_code(exc)
                error_key = f"calendar_fetch_disk_{status_code or 'unknown'}"
                self._log_calendar_warning_once(
                    now,
                    f"News fetch failed, using disk cache: {exc}",
                    key=error_key,
                )
                self._cached_events = disk_cached
                self._cache_expires_at = max(
                    now + timedelta(seconds=self.settings.news.cache_ttl_seconds),
                    self._calendar_backoff_until or now,
                )
                return disk_cached
            raise RuntimeError(f"news fetch failed: {exc}") from exc

    def symbol_currencies(self, symbol: str) -> list[str]:
        normalized = symbol.upper()
        configured = self.settings.news.symbol_currency_map.get(normalized)
        if configured:
            return configured

        inferred: list[str] = []
        if len(normalized) >= 6:
            head = normalized[:3]
            tail = normalized[3:6]
            if head in KNOWN_CURRENCIES:
                inferred.append(head)
            if tail in KNOWN_CURRENCIES and tail not in inferred:
                inferred.append(tail)
        if inferred:
            return inferred

        return SYMBOL_CURRENCY_FALLBACKS.get(normalized, ["USD"])

    def relevant_events(self, symbol: str, now: datetime | None = None) -> list[NewsEvent]:
        if not self.settings.news.enabled:
            return []

        current_time = now or datetime.now(timezone.utc)
        lower_bound = current_time - timedelta(minutes=self.settings.news.lookback_minutes)
        upper_bound = current_time + timedelta(minutes=self.settings.news.lookahead_minutes)
        currencies = set(self.symbol_currencies(symbol))

        return [
            event
            for event in self._events(current_time)
            if event.currency in currencies
            and event.importance >= self.settings.news.min_importance
            and lower_bound <= event.timestamp <= upper_bound
        ]

    def evaluate(self, symbol: str, now: datetime | None = None) -> NewsGuardDecision:
        if not self.settings.news.enabled:
            return NewsGuardDecision(
                symbol=symbol,
                blocked=False,
                close_positions=False,
                reason="news_filter_off",
                phase="off",
                event_tag="SYSTEM",
                event_family="SYSTEM",
            )

        current_time = now or datetime.now(timezone.utc)

        crypto_decision = self._crypto_sentiment_guard(symbol, current_time)
        if crypto_decision is not None:
            return crypto_decision

        try:
            events = self.relevant_events(symbol, current_time)
        except Exception as exc:
            logger.warning("News guard failed for {}: {}", symbol, exc)
            if self.settings.news.fail_closed:
                return NewsGuardDecision(
                    symbol=symbol,
                    blocked=True,
                    close_positions=False,
                    reason="news_feed_unavailable",
                    phase="feed_unavailable",
                    event_tag="SYSTEM",
                    event_family="SYSTEM",
                    strict_spread_limit_points=self.settings.risk.news_spread_limit_points,
                )
            return NewsGuardDecision(
                symbol=symbol,
                blocked=False,
                close_positions=False,
                reason="news_feed_unavailable_fail_open",
                phase="feed_unavailable",
                event_tag="SYSTEM",
                event_family="SYSTEM",
            )

        if not events:
            return NewsGuardDecision(
                symbol=symbol,
                blocked=False,
                close_positions=False,
                reason="no_relevant_news",
                phase="none",
                event_tag="NONE",
                event_family="NONE",
            )

        ranked = [
            self._decision_for_event(symbol=symbol, event=event, current_time=current_time)
            for event in events
        ]
        _, _, _, selected = sorted(ranked, key=lambda item: (-item[0], -item[1], item[2]))[0]
        return selected

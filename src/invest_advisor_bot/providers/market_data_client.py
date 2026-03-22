from __future__ import annotations

import asyncio
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

import pandas as pd
import httpx
import yfinance as yf
from cachetools import TTLCache
from loguru import logger

from invest_advisor_bot.universe import (
    INDEX_UNIVERSE_SOURCES,
    StockUniverseMember,
    US_LARGE_CAP_STOCK_UNIVERSE,
    normalize_ticker_for_market_data,
)

DEFAULT_ASSET_UNIVERSE: dict[str, str] = {
    "gold_futures": "GC=F",
    "sp500_index": "^GSPC",
    "nasdaq_index": "^IXIC",
    "spy_etf": "SPY",
    "qqq_etf": "QQQ",
    "gld_etf": "GLD",
    "iau_etf": "IAU",
    "vti_etf": "VTI",
    "xlf_etf": "XLF",
    "xle_etf": "XLE",
    "xlk_etf": "XLK",
    "xly_etf": "XLY",
    "xlp_etf": "XLP",
    "xlv_etf": "XLV",
    "xli_etf": "XLI",
    "xlb_etf": "XLB",
    "xlu_etf": "XLU",
    "xlc_etf": "XLC",
    "xlre_etf": "XLRE",
    "tlt_etf": "TLT",
    "voo_etf": "VOO",
}


@dataclass(slots=True, frozen=True)
class AssetQuote:
    ticker: str
    name: str
    currency: str | None
    exchange: str | None
    price: float
    previous_close: float | None
    open_price: float | None
    day_high: float | None
    day_low: float | None
    volume: int | None
    timestamp: datetime | None


@dataclass(slots=True, frozen=True)
class OhlcvBar:
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(slots=True, frozen=True)
class StockFundamentals:
    ticker: str
    company_name: str
    sector: str | None
    industry: str | None
    market_cap: float | None
    trailing_pe: float | None
    forward_pe: float | None
    price_to_book: float | None
    dividend_yield: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    profit_margin: float | None
    operating_margin: float | None
    return_on_equity: float | None
    debt_to_equity: float | None
    analyst_target_price: float | None
    free_cash_flow: float | None = None
    free_cash_flow_margin: float | None = None
    revenue_qoq_change: float | None = None
    operating_margin_qoq_change: float | None = None
    free_cash_flow_qoq_change: float | None = None
    revenue_quality_trend: str | None = None
    margin_quality_trend: str | None = None
    fcf_quality_trend: str | None = None


@dataclass(slots=True, frozen=True)
class EarningsEvent:
    ticker: str
    earnings_at: datetime
    eps_estimate: float | None
    reported_eps: float | None
    surprise_pct: float | None


@dataclass(slots=True, frozen=True)
class RecentEarningsResult:
    ticker: str
    earnings_at: datetime
    eps_estimate: float | None
    reported_eps: float | None
    surprise_pct: float | None


@dataclass(slots=True, frozen=True)
class AnalystExpectationProfile:
    ticker: str
    revenue_growth_estimate_current_q: float | None
    eps_growth_estimate_current_q: float | None
    revenue_analyst_count: int | None
    eps_analyst_count: int | None


class MarketDataClient:
    """Async wrapper around yfinance for quote and OHLCV retrieval."""

    def __init__(
        self,
        asset_universe: Mapping[str, str] | None = None,
        *,
        cache_ttl_seconds: int = 900,
        cache_maxsize: int = 256,
        alpha_vantage_api_key: str = "",
        provider_order: Sequence[str] | None = None,
        http_timeout_seconds: float = 12.0,
    ) -> None:
        self.asset_universe: dict[str, str] = dict(asset_universe or DEFAULT_ASSET_UNIVERSE)
        self.alpha_vantage_api_key = alpha_vantage_api_key.strip()
        normalized_order = tuple(
            item.strip().casefold()
            for item in (provider_order or ("alpha_vantage", "yfinance"))
            if item and item.strip()
        )
        if normalized_order:
            self.provider_order = normalized_order
        elif self.alpha_vantage_api_key:
            self.provider_order = ("alpha_vantage", "yfinance")
        else:
            self.provider_order = ("yfinance",)
        self.http_timeout_seconds = max(2.0, float(http_timeout_seconds))
        self._cache_lock = RLock()
        self._http_client_lock = RLock()
        self._http_client: httpx.Client | None = None
        self._latest_price_cache: TTLCache[tuple[str, str], AssetQuote | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )
        self._history_cache: TTLCache[tuple[str, str, str, int | None], list[OhlcvBar]] = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )
        self._snapshot_cache: TTLCache[tuple[str, ...], dict[str, AssetQuote | None]] = TTLCache(
            maxsize=32,
            ttl=cache_ttl_seconds,
        )
        self._core_history_cache: TTLCache[tuple[str, str, int | None, tuple[str, ...]], dict[str, list[OhlcvBar]]] = (
            TTLCache(maxsize=32, ttl=cache_ttl_seconds)
        )
        self._macro_context_cache: TTLCache[str, dict[str, float | None]] = TTLCache(
            maxsize=8,
            ttl=cache_ttl_seconds,
        )
        self._fundamentals_cache: TTLCache[str, StockFundamentals | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )
        self._stock_snapshot_cache: TTLCache[tuple[str, ...], dict[str, AssetQuote | None]] = TTLCache(
            maxsize=16,
            ttl=cache_ttl_seconds,
        )
        self._stock_history_cache: TTLCache[tuple[str, str, int | None], dict[str, list[OhlcvBar]]] = TTLCache(
            maxsize=16,
            ttl=cache_ttl_seconds,
        )
        self._stock_fundamentals_batch_cache: TTLCache[tuple[str, ...], dict[str, StockFundamentals | None]] = TTLCache(
            maxsize=16,
            ttl=cache_ttl_seconds,
        )
        self._earnings_cache: TTLCache[str, EarningsEvent | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )
        self._recent_earnings_cache: TTLCache[tuple[str, int], RecentEarningsResult | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )
        self._expectation_cache: TTLCache[str, AnalystExpectationProfile | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=cache_ttl_seconds,
        )
        self._dynamic_universe_cache: TTLCache[tuple[tuple[str, ...], int | None], dict[str, StockUniverseMember]] = TTLCache(
            maxsize=8,
            ttl=max(cache_ttl_seconds, 3600),
        )

    async def aclose(self) -> None:
        with self._http_client_lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            client.close()

    async def get_latest_price(self, ticker: str) -> AssetQuote | None:
        cache_key = ("latest", ticker.upper())
        with self._cache_lock:
            if cache_key in self._latest_price_cache:
                return self._latest_price_cache[cache_key]

        try:
            result = await asyncio.to_thread(self._get_latest_price_sync, ticker)
        except Exception as exc:
            logger.exception("Failed to fetch latest price for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._latest_price_cache[cache_key] = result
        return result

    async def get_latest_prices(self, tickers: Sequence[str]) -> dict[str, AssetQuote | None]:
        tasks = [self.get_latest_price(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        payload: dict[str, AssetQuote | None] = {}
        for ticker, result in zip(tickers, results, strict=False):
            if isinstance(result, Exception):
                logger.exception("Latest price task failed for {}: {}", ticker, result)
                payload[ticker] = None
                continue
            payload[ticker] = result
        return payload

    async def get_history(
        self,
        ticker: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
        limit: int | None = None,
    ) -> list[OhlcvBar]:
        cache_key = (ticker.upper(), period, interval, limit)
        with self._cache_lock:
            if cache_key in self._history_cache:
                return list(self._history_cache[cache_key] or [])

        try:
            result = await asyncio.to_thread(
                self._get_history_sync,
                ticker,
                period,
                interval,
                limit,
            )
        except Exception as exc:
            logger.exception(
                "Failed to fetch history for {} (period={}, interval={}): {}",
                ticker,
                period,
                interval,
                exc,
            )
            return []
        with self._cache_lock:
            self._history_cache[cache_key] = list(result)
        return result

    async def get_core_market_snapshot(self) -> dict[str, AssetQuote | None]:
        cache_key = tuple(self.asset_universe.items())
        with self._cache_lock:
            if cache_key in self._snapshot_cache:
                return dict(self._snapshot_cache[cache_key] or {})

        tickers = list(self.asset_universe.values())
        quotes = await self.get_latest_prices(tickers)
        payload = {asset_name: quotes.get(ticker) for asset_name, ticker in self.asset_universe.items()}
        with self._cache_lock:
            self._snapshot_cache[cache_key] = dict(payload)
        return payload

    async def get_core_market_history(
        self,
        *,
        period: str = "6mo",
        interval: str = "1d",
        limit: int | None = None,
    ) -> dict[str, list[OhlcvBar]]:
        cache_key = (period, interval, limit, tuple(self.asset_universe.items()))
        with self._cache_lock:
            if cache_key in self._core_history_cache:
                return {asset: list(bars) for asset, bars in (self._core_history_cache[cache_key] or {}).items()}

        tasks = {
            asset_name: self.get_history(ticker, period=period, interval=interval, limit=limit)
            for asset_name, ticker in self.asset_universe.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        payload: dict[str, list[OhlcvBar]] = {}
        for asset_name, result in zip(tasks.keys(), results, strict=False):
            if isinstance(result, Exception):
                logger.exception("History task failed for {}: {}", asset_name, result)
                payload[asset_name] = []
                continue
            payload[asset_name] = result
        with self._cache_lock:
            self._core_history_cache[cache_key] = {asset: list(bars) for asset, bars in payload.items()}
        return payload

    async def get_fundamentals(self, ticker: str) -> StockFundamentals | None:
        cache_key = ticker.upper()
        with self._cache_lock:
            if cache_key in self._fundamentals_cache:
                return self._fundamentals_cache[cache_key]

        try:
            result = await asyncio.to_thread(self._get_fundamentals_sync, ticker)
        except Exception as exc:
            logger.exception("Failed to fetch fundamentals for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._fundamentals_cache[cache_key] = result
        return result

    async def get_stock_universe_snapshot(
        self,
        stock_universe: Mapping[str, StockUniverseMember] | None = None,
    ) -> dict[str, AssetQuote | None]:
        universe = dict(stock_universe or US_LARGE_CAP_STOCK_UNIVERSE)
        cache_key = tuple((alias, member.ticker) for alias, member in universe.items())
        with self._cache_lock:
            if cache_key in self._stock_snapshot_cache:
                return dict(self._stock_snapshot_cache[cache_key] or {})

        quotes = await self.get_latest_prices([member.ticker for member in universe.values()])
        payload = {alias: quotes.get(member.ticker) for alias, member in universe.items()}
        with self._cache_lock:
            self._stock_snapshot_cache[cache_key] = dict(payload)
        return payload

    async def get_dynamic_stock_universe(
        self,
        *,
        indexes: Sequence[str] = ("sp500", "nasdaq100"),
        max_members: int | None = None,
    ) -> dict[str, StockUniverseMember]:
        normalized_indexes = tuple(dict.fromkeys(index.strip().casefold() for index in indexes if index.strip()))
        cache_key = (normalized_indexes, max_members)
        with self._cache_lock:
            if cache_key in self._dynamic_universe_cache:
                return dict(self._dynamic_universe_cache[cache_key] or {})
        result = await asyncio.to_thread(self._load_dynamic_stock_universe_sync, normalized_indexes, max_members)
        with self._cache_lock:
            self._dynamic_universe_cache[cache_key] = dict(result)
        return result

    async def get_stock_universe_history(
        self,
        *,
        stock_universe: Mapping[str, StockUniverseMember] | None = None,
        period: str = "6mo",
        interval: str = "1d",
        limit: int | None = None,
    ) -> dict[str, list[OhlcvBar]]:
        universe = dict(stock_universe or US_LARGE_CAP_STOCK_UNIVERSE)
        cache_key = (period, interval, limit, tuple((alias, member.ticker) for alias, member in universe.items()))
        with self._cache_lock:
            if cache_key in self._stock_history_cache:
                return {alias: list(bars) for alias, bars in (self._stock_history_cache[cache_key] or {}).items()}

        tasks = {
            alias: self.get_history(member.ticker, period=period, interval=interval, limit=limit)
            for alias, member in universe.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, list[OhlcvBar]] = {}
        for alias, result in zip(tasks.keys(), results, strict=False):
            payload[alias] = [] if isinstance(result, Exception) else result
        with self._cache_lock:
            self._stock_history_cache[cache_key] = {alias: list(bars) for alias, bars in payload.items()}
        return payload

    async def get_stock_universe_fundamentals(
        self,
        stock_universe: Mapping[str, StockUniverseMember] | None = None,
    ) -> dict[str, StockFundamentals | None]:
        universe = dict(stock_universe or US_LARGE_CAP_STOCK_UNIVERSE)
        cache_key = tuple((alias, member.ticker) for alias, member in universe.items())
        with self._cache_lock:
            if cache_key in self._stock_fundamentals_batch_cache:
                return dict(self._stock_fundamentals_batch_cache[cache_key] or {})

        tasks = {alias: self.get_fundamentals(member.ticker) for alias, member in universe.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, StockFundamentals | None] = {}
        for alias, result in zip(tasks.keys(), results, strict=False):
            payload[alias] = None if isinstance(result, Exception) else result
        with self._cache_lock:
            self._stock_fundamentals_batch_cache[cache_key] = dict(payload)
        return payload

    async def get_next_earnings_event(self, ticker: str) -> EarningsEvent | None:
        cache_key = ticker.upper()
        with self._cache_lock:
            if cache_key in self._earnings_cache:
                return self._earnings_cache[cache_key]
        try:
            result = await asyncio.to_thread(self._get_next_earnings_event_sync, ticker)
        except Exception as exc:
            logger.exception("Failed to fetch earnings event for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._earnings_cache[cache_key] = result
        return result

    async def get_earnings_calendar(
        self,
        tickers: Sequence[str],
        *,
        days_ahead: int = 7,
    ) -> dict[str, EarningsEvent]:
        tasks = {ticker.upper(): self.get_next_earnings_event(ticker) for ticker in tickers if ticker.strip()}
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        now = datetime.now(timezone.utc)
        payload: dict[str, EarningsEvent] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            if isinstance(result, Exception) or result is None:
                continue
            if result.earnings_at < now:
                continue
            if (result.earnings_at - now).days > max(0, days_ahead):
                continue
            payload[ticker] = result
        return payload

    async def get_recent_earnings_results(
        self,
        tickers: Sequence[str],
        *,
        lookback_days: int = 7,
    ) -> dict[str, RecentEarningsResult]:
        tasks = {
            ticker.upper(): self.get_recent_earnings_result(ticker, lookback_days=lookback_days)
            for ticker in tickers
            if ticker.strip()
        }
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, RecentEarningsResult] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            if isinstance(result, Exception) or result is None:
                continue
            payload[ticker] = result
        return payload

    async def get_recent_earnings_result(self, ticker: str, *, lookback_days: int = 7) -> RecentEarningsResult | None:
        cache_key = (ticker.upper(), max(1, lookback_days))
        with self._cache_lock:
            if cache_key in self._recent_earnings_cache:
                return self._recent_earnings_cache[cache_key]
        try:
            result = await asyncio.to_thread(self._get_recent_earnings_result_sync, ticker, lookback_days)
        except Exception as exc:
            logger.exception("Failed to fetch recent earnings result for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._recent_earnings_cache[cache_key] = result
        return result

    async def get_analyst_expectation_profile(self, ticker: str) -> AnalystExpectationProfile | None:
        cache_key = ticker.upper()
        with self._cache_lock:
            if cache_key in self._expectation_cache:
                return self._expectation_cache[cache_key]
        try:
            result = await asyncio.to_thread(self._get_analyst_expectation_profile_sync, ticker)
        except Exception as exc:
            logger.exception("Failed to fetch analyst expectation profile for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._expectation_cache[cache_key] = result
        return result

    async def get_analyst_expectation_profiles(self, tickers: Sequence[str]) -> dict[str, AnalystExpectationProfile | None]:
        tasks = {ticker.upper(): self.get_analyst_expectation_profile(ticker) for ticker in tickers if ticker.strip()}
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, AnalystExpectationProfile | None] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            payload[ticker] = None if isinstance(result, Exception) else result
        return payload

    def _get_latest_price_sync(self, ticker: str) -> AssetQuote | None:
        providers = self._effective_provider_order()
        for provider_name in providers:
            if provider_name == "alpha_vantage":
                quote = self._get_latest_price_alpha_vantage_sync(ticker)
                if quote is not None:
                    return quote
                continue
            if provider_name == "yfinance":
                quote = self._get_latest_price_yfinance_sync(ticker)
                if quote is not None:
                    return quote
        return None

    def _get_latest_price_yfinance_sync(self, ticker: str) -> AssetQuote | None:
        ticker_client = yf.Ticker(ticker)

        fast_info: Mapping[str, Any] = {}
        try:
            fast_info = ticker_client.fast_info or {}
        except Exception as exc:
            logger.warning("fast_info unavailable for {}: {}", ticker, exc)

        history = self._fetch_history_frame(ticker_client, period="5d", interval="1d")
        if history.empty:
            logger.warning("No recent history returned for {}", ticker)
            return None

        last_row = history.iloc[-1]
        metadata = self._safe_history_metadata(history)

        price = self._as_float(fast_info.get("lastPrice")) or self._as_float(last_row.get("Close"))
        if price is None:
            logger.warning("No usable latest price for {}", ticker)
            return None

        previous_close = self._as_float(fast_info.get("previousClose"))
        if previous_close is None and len(history) > 1:
            previous_close = self._as_float(history.iloc[-2].get("Close"))

        timestamp = self._normalize_timestamp(history.index[-1])

        return AssetQuote(
            ticker=ticker,
            name=str(metadata.get("shortName") or metadata.get("longName") or ticker),
            currency=self._as_optional_str(fast_info.get("currency") or metadata.get("currency")),
            exchange=self._as_optional_str(fast_info.get("exchange") or metadata.get("exchangeName")),
            price=price,
            previous_close=previous_close,
            open_price=self._as_float(fast_info.get("open")) or self._as_float(last_row.get("Open")),
            day_high=self._as_float(fast_info.get("dayHigh")) or self._as_float(last_row.get("High")),
            day_low=self._as_float(fast_info.get("dayLow")) or self._as_float(last_row.get("Low")),
            volume=self._as_int(fast_info.get("lastVolume")) or self._as_int(last_row.get("Volume")),
            timestamp=timestamp,
        )

    def _get_latest_price_alpha_vantage_sync(self, ticker: str) -> AssetQuote | None:
        symbol = self._normalize_alpha_vantage_symbol(ticker)
        if symbol is None:
            return None
        payload = self._fetch_alpha_vantage_json(function="GLOBAL_QUOTE", symbol=symbol)
        if not isinstance(payload, Mapping):
            return None
        quote = payload.get("Global Quote")
        if not isinstance(quote, Mapping):
            return None
        price = self._as_float(quote.get("05. price"))
        if price is None:
            return None
        timestamp_raw = self._as_optional_str(quote.get("07. latest trading day"))
        timestamp = None
        if timestamp_raw:
            try:
                timestamp = datetime.fromisoformat(timestamp_raw).replace(tzinfo=timezone.utc)
            except ValueError:
                timestamp = None
        return AssetQuote(
            ticker=ticker.upper(),
            name=ticker.upper(),
            currency="USD",
            exchange=None,
            price=price,
            previous_close=self._as_float(quote.get("08. previous close")),
            open_price=self._as_float(quote.get("02. open")),
            day_high=self._as_float(quote.get("03. high")),
            day_low=self._as_float(quote.get("04. low")),
            volume=self._as_int(quote.get("06. volume")),
            timestamp=timestamp,
        )

    def _get_history_sync(
        self,
        ticker: str,
        period: str,
        interval: str,
        limit: int | None,
    ) -> list[OhlcvBar]:
        providers = self._effective_provider_order()
        for provider_name in providers:
            if provider_name == "alpha_vantage":
                bars = self._get_history_alpha_vantage_sync(ticker, period=period, interval=interval, limit=limit)
                if bars:
                    return bars
                continue
            if provider_name == "yfinance":
                bars = self._get_history_yfinance_sync(ticker, period=period, interval=interval, limit=limit)
                if bars:
                    return bars
        return []

    def _get_history_yfinance_sync(
        self,
        ticker: str,
        period: str,
        interval: str,
        limit: int | None,
    ) -> list[OhlcvBar]:
        ticker_client = yf.Ticker(ticker)
        history = self._fetch_history_frame(ticker_client, period=period, interval=interval)
        if history.empty:
            logger.warning("No OHLCV history returned for {}", ticker)
            return []

        frame = history.tail(limit).copy() if limit is not None and limit > 0 else history.copy()
        bars: list[OhlcvBar] = []

        for index, row in frame.iterrows():
            try:
                timestamp = self._normalize_timestamp(index)
                if timestamp is None:
                    continue

                bars.append(
                    OhlcvBar(
                        ticker=ticker,
                        timestamp=timestamp,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(float(row.get("Volume", 0) or 0)),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed OHLCV row for {}: {}", ticker, exc)
        return bars

    def _get_history_alpha_vantage_sync(
        self,
        ticker: str,
        *,
        period: str,
        interval: str,
        limit: int | None,
    ) -> list[OhlcvBar]:
        if interval.strip().casefold() != "1d":
            return []
        symbol = self._normalize_alpha_vantage_symbol(ticker)
        if symbol is None:
            return []
        outputsize = "compact"
        if limit is None or limit > 100 or (self._period_to_days(period) or 0) > 100:
            outputsize = "full"
        payload = self._fetch_alpha_vantage_json(
            function="TIME_SERIES_DAILY_ADJUSTED",
            symbol=symbol,
            extra_params={"outputsize": outputsize},
        )
        if not isinstance(payload, Mapping):
            return []
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, Mapping):
            return []
        cutoff = self._period_cutoff(period)
        rows: list[OhlcvBar] = []
        for date_key, values in series.items():
            if not isinstance(date_key, str) or not isinstance(values, Mapping):
                continue
            try:
                timestamp = datetime.fromisoformat(date_key).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if cutoff is not None and timestamp < cutoff:
                continue
            close_value = self._as_float(values.get("4. close")) or self._as_float(values.get("5. adjusted close"))
            open_value = self._as_float(values.get("1. open"))
            high_value = self._as_float(values.get("2. high"))
            low_value = self._as_float(values.get("3. low"))
            volume_value = self._as_int(values.get("6. volume"))
            if None in {open_value, high_value, low_value, close_value}:
                continue
            rows.append(
                OhlcvBar(
                    ticker=ticker.upper(),
                    timestamp=timestamp,
                    open=float(open_value),
                    high=float(high_value),
                    low=float(low_value),
                    close=float(close_value),
                    volume=int(volume_value or 0),
                )
            )
        rows.sort(key=lambda item: item.timestamp)
        if limit is not None and limit > 0:
            return rows[-limit:]
        return rows

    def _effective_provider_order(self) -> tuple[str, ...]:
        providers: list[str] = []
        for provider_name in self.provider_order:
            if provider_name == "alpha_vantage" and not self.alpha_vantage_api_key:
                continue
            if provider_name in {"alpha_vantage", "yfinance"} and provider_name not in providers:
                providers.append(provider_name)
        if "yfinance" not in providers:
            providers.append("yfinance")
        return tuple(providers)

    def _fetch_alpha_vantage_json(
        self,
        *,
        function: str,
        symbol: str,
        extra_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.alpha_vantage_api_key:
            return None
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.alpha_vantage_api_key,
        }
        if extra_params:
            params.update(dict(extra_params))
        try:
            response = self._get_http_client().get(
                "https://www.alphavantage.co/query",
                params=params,
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Alpha Vantage request failed for {} ({}): {}", symbol, function, exc)
            return None
        if not isinstance(payload, Mapping):
            return None
        note = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
        if note:
            logger.warning("Alpha Vantage returned non-data response for {} ({}): {}", symbol, function, note)
            return None
        return dict(payload)

    def _get_http_client(self) -> httpx.Client:
        with self._http_client_lock:
            client = self._http_client
            if client is None:
                client = httpx.Client(timeout=self.http_timeout_seconds)
                self._http_client = client
            return client

    @staticmethod
    def _normalize_alpha_vantage_symbol(ticker: str) -> str | None:
        normalized = ticker.strip().upper()
        if not normalized or "^" in normalized or "=" in normalized:
            return None
        return normalized.replace("-", ".")

    @staticmethod
    def _period_to_days(period: str) -> int | None:
        normalized = period.strip().casefold()
        if not normalized or normalized == "max":
            return None
        if normalized == "ytd":
            today = datetime.now(timezone.utc).date()
            start = datetime(today.year, 1, 1, tzinfo=timezone.utc).date()
            return max(1, (today - start).days)
        units = (
            ("day", 1),
            ("days", 1),
            ("d", 1),
            ("wk", 7),
            ("w", 7),
            ("mo", 31),
            ("m", 31),
            ("y", 366),
        )
        for suffix, multiplier in units:
            if not normalized.endswith(suffix):
                continue
            number_text = normalized[: -len(suffix)]
            if not number_text.isdigit():
                return None
            return int(number_text) * multiplier
        return None

    @classmethod
    def _period_cutoff(cls, period: str) -> datetime | None:
        days = cls._period_to_days(period)
        if days is None:
            return None
        return datetime.now(timezone.utc) - timedelta(days=max(1, days))

    @staticmethod
    def _fetch_history_frame(
        ticker_client: yf.Ticker,
        *,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        try:
            history = ticker_client.history(
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:
            logger.warning(
                "yfinance history request failed for {} (period={}, interval={}): {}",
                ticker_client.ticker,
                period,
                interval,
                exc,
            )
            return pd.DataFrame()

        if history is None or history.empty:
            return pd.DataFrame()

        frame = history.reset_index(drop=False)
        timestamp_column = "Datetime" if "Datetime" in frame.columns else "Date"
        frame = frame.rename(columns={timestamp_column: "Timestamp"})
        frame = frame.set_index("Timestamp")
        return frame.sort_index()

    @staticmethod
    def _safe_history_metadata(history: pd.DataFrame) -> Mapping[str, Any]:
        metadata = history.attrs.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}

    @staticmethod
    def _normalize_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone.utc)
        else:
            timestamp = timestamp.tz_convert(timezone.utc)
        return timestamp.to_pydatetime()

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    async def get_macro_context(self) -> dict[str, float | None]:
        """Fetch VIX, 10Y Yield, and CPI Inflation Rate."""
        cache_key = "macro_context"
        with self._cache_lock:
            if cache_key in self._macro_context_cache:
                return dict(self._macro_context_cache[cache_key])
        result = await asyncio.to_thread(self._get_macro_context_sync)
        with self._cache_lock:
            self._macro_context_cache[cache_key] = dict(result)
        return result

    def _get_macro_context_sync(self) -> dict[str, float | None]:
        vix = None
        tnx = None
        cpi_yoy = None

        try:
            vix_ticker = yf.Ticker("^VIX")
            vix = vix_ticker.fast_info.get("lastPrice")
        except Exception as e:
            logger.warning("Failed to fetch VIX: {}", e)

        try:
            tnx_ticker = yf.Ticker("^TNX")
            tnx = tnx_ticker.fast_info.get("lastPrice")
        except Exception as e:
            logger.warning("Failed to fetch TNX: {}", e)

        try:
            response = self._get_http_client().get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
                headers={"User-Agent": "invest-advisor-bot/0.1"},
                follow_redirects=True,
            )
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
            date_column = "DATE" if "DATE" in frame.columns else "observation_date" if "observation_date" in frame.columns else None
            if date_column is not None and "CPIAUCSL" in frame.columns:
                frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
                frame["CPIAUCSL"] = pd.to_numeric(frame["CPIAUCSL"], errors="coerce")
                frame = frame.dropna(subset=[date_column, "CPIAUCSL"]).sort_values(date_column)
                if len(frame) >= 13:
                    latest = float(frame.iloc[-1]["CPIAUCSL"])
                    one_year_ago = float(frame.iloc[-13]["CPIAUCSL"])
                    if one_year_ago != 0:
                        cpi_yoy = round(((latest - one_year_ago) / one_year_ago) * 100.0, 2)
        except Exception as e:
            logger.warning("Failed to fetch CPI from FRED CSV: {}", e)

        return {
            "vix": float(vix) if vix else None,
            "tnx": float(tnx) if tnx else None,
            "cpi_yoy": cpi_yoy,
        }

    def _get_fundamentals_sync(self, ticker: str) -> StockFundamentals | None:
        ticker_client = yf.Ticker(ticker)
        try:
            info = ticker_client.info or {}
        except Exception as exc:
            logger.warning("yfinance info unavailable for {}: {}", ticker, exc)
            return None
        if not isinstance(info, Mapping) or not info:
            return None
        cashflow_frame, income_frame = self._load_financial_statement_frames(ticker_client)
        quarterly_quality = self._extract_quarterly_quality_profile(cashflow_frame, income_frame)
        return StockFundamentals(
            ticker=ticker.upper(),
            company_name=str(info.get("shortName") or info.get("longName") or ticker).strip(),
            sector=self._as_optional_str(info.get("sector")),
            industry=self._as_optional_str(info.get("industry")),
            market_cap=self._as_float(info.get("marketCap")),
            trailing_pe=self._as_float(info.get("trailingPE")),
            forward_pe=self._as_float(info.get("forwardPE")),
            price_to_book=self._as_float(info.get("priceToBook")),
            dividend_yield=self._as_float(info.get("dividendYield")),
            revenue_growth=self._as_float(info.get("revenueGrowth")),
            earnings_growth=self._as_float(info.get("earningsGrowth")),
            profit_margin=self._as_float(info.get("profitMargins")),
            operating_margin=self._as_float(info.get("operatingMargins")),
            return_on_equity=self._as_float(info.get("returnOnEquity")),
            debt_to_equity=self._as_float(info.get("debtToEquity")),
            analyst_target_price=self._as_float(info.get("targetMeanPrice")),
            free_cash_flow=quarterly_quality["free_cash_flow"],
            free_cash_flow_margin=quarterly_quality["free_cash_flow_margin"],
            revenue_qoq_change=quarterly_quality["revenue_qoq_change"],
            operating_margin_qoq_change=quarterly_quality["operating_margin_qoq_change"],
            free_cash_flow_qoq_change=quarterly_quality["free_cash_flow_qoq_change"],
            revenue_quality_trend=quarterly_quality["revenue_quality_trend"],
            margin_quality_trend=quarterly_quality["margin_quality_trend"],
            fcf_quality_trend=quarterly_quality["fcf_quality_trend"],
        )

    def _get_next_earnings_event_sync(self, ticker: str) -> EarningsEvent | None:
        ticker_client = yf.Ticker(ticker)
        try:
            dates = ticker_client.get_earnings_dates(limit=4)
        except Exception as exc:
            logger.warning("Failed to fetch earnings dates for {}: {}", ticker, exc)
            return None
        if dates is None or dates.empty:
            return None
        frame = dates.reset_index()
        timestamp_column = frame.columns[0]
        now = datetime.now(timezone.utc)
        for _, row in frame.iterrows():
            earnings_at = self._normalize_timestamp(row.get(timestamp_column))
            if earnings_at is None or earnings_at < now:
                continue
            return EarningsEvent(
                ticker=ticker.upper(),
                earnings_at=earnings_at,
                eps_estimate=self._as_float(row.get("EPS Estimate")),
                reported_eps=self._as_float(row.get("Reported EPS")),
                surprise_pct=self._as_float(row.get("Surprise(%)")),
            )
        return None

    def _get_recent_earnings_result_sync(self, ticker: str, lookback_days: int) -> RecentEarningsResult | None:
        ticker_client = yf.Ticker(ticker)
        try:
            dates = ticker_client.get_earnings_dates(limit=8)
        except Exception as exc:
            logger.warning("Failed to fetch recent earnings dates for {}: {}", ticker, exc)
            return None
        if dates is None or dates.empty:
            return None
        frame = dates.reset_index()
        timestamp_column = frame.columns[0]
        now = datetime.now(timezone.utc)
        lookback_window = max(1, lookback_days)
        for _, row in frame.iterrows():
            earnings_at = self._normalize_timestamp(row.get(timestamp_column))
            if earnings_at is None or earnings_at > now:
                continue
            if (now - earnings_at).days > lookback_window:
                continue
            return RecentEarningsResult(
                ticker=ticker.upper(),
                earnings_at=earnings_at,
                eps_estimate=self._as_float(row.get("EPS Estimate")),
                reported_eps=self._as_float(row.get("Reported EPS")),
                surprise_pct=self._as_float(row.get("Surprise(%)")),
            )
        return None

    def _get_analyst_expectation_profile_sync(self, ticker: str) -> AnalystExpectationProfile | None:
        ticker_client = yf.Ticker(ticker)
        revenue_growth = None
        revenue_analyst_count = None
        eps_growth = None
        eps_analyst_count = None
        try:
            revenue_frame = ticker_client.get_revenue_estimate()
            if revenue_frame is not None and not revenue_frame.empty and "0q" in revenue_frame.index:
                row = revenue_frame.loc["0q"]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                revenue_growth = self._as_float(row.get("growth"))
                revenue_analyst_count = self._as_int(row.get("numberOfAnalysts"))
        except Exception as exc:
            logger.warning("Revenue estimate unavailable for {}: {}", ticker, exc)
        try:
            earnings_frame = ticker_client.get_earnings_estimate()
            if earnings_frame is not None and not earnings_frame.empty and "0q" in earnings_frame.index:
                row = earnings_frame.loc["0q"]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                eps_growth = self._as_float(row.get("growth"))
                eps_analyst_count = self._as_int(row.get("numberOfAnalysts"))
        except Exception as exc:
            logger.warning("Earnings estimate unavailable for {}: {}", ticker, exc)
        if all(value is None for value in (revenue_growth, revenue_analyst_count, eps_growth, eps_analyst_count)):
            return None
        return AnalystExpectationProfile(
            ticker=ticker.upper(),
            revenue_growth_estimate_current_q=revenue_growth,
            eps_growth_estimate_current_q=eps_growth,
            revenue_analyst_count=revenue_analyst_count,
            eps_analyst_count=eps_analyst_count,
        )

    def _load_dynamic_stock_universe_sync(
        self,
        indexes: Sequence[str],
        max_members: int | None,
    ) -> dict[str, StockUniverseMember]:
        members: dict[str, StockUniverseMember] = {}
        for index_name in indexes:
            config = INDEX_UNIVERSE_SOURCES.get(index_name)
            if config is None:
                continue
            try:
                response = self._get_http_client().get(
                    config["url"],
                    headers={"User-Agent": "Mozilla/5.0 (compatible; InvestAdvisorBot/1.0)"},
                    follow_redirects=True,
                )
                response.raise_for_status()
                tables = pd.read_html(StringIO(response.text))
            except Exception as exc:
                logger.warning("Failed to load {} universe from {}: {}", index_name, config["url"], exc)
                continue
            for table in tables:
                if config["ticker_column"] not in table.columns:
                    continue
                for _, row in table.iterrows():
                    ticker = normalize_ticker_for_market_data(str(row.get(config["ticker_column"]) or ""))
                    company_name = str(row.get(config["name_column"]) or ticker).strip()
                    sector = str(row.get(config["sector_column"]) or "Unknown").strip() or "Unknown"
                    if not ticker:
                        continue
                    alias = ticker.casefold().replace("-", "_")
                    members.setdefault(
                        alias,
                        StockUniverseMember(
                            ticker=ticker,
                            company_name=company_name,
                            sector=sector,
                            benchmark=index_name,
                        ),
                    )
                if members:
                    break
        if not members:
            members = dict(US_LARGE_CAP_STOCK_UNIVERSE)
        if max_members is not None and max_members > 0:
            limited_items = list(members.items())[:max_members]
            return dict(limited_items)
        return members

    def _load_financial_statement_frames(self, ticker_client: yf.Ticker) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            cashflow_frame = ticker_client.quarterly_cashflow
            if cashflow_frame is None or cashflow_frame.empty:
                cashflow_frame = ticker_client.cashflow
        except Exception as exc:
            logger.warning("Cash flow statement unavailable for {}: {}", ticker_client.ticker, exc)
            cashflow_frame = pd.DataFrame()
        try:
            income_frame = ticker_client.quarterly_income_stmt
            if income_frame is None or income_frame.empty:
                income_frame = ticker_client.income_stmt
        except Exception as exc:
            logger.warning("Income statement unavailable for {}: {}", ticker_client.ticker, exc)
            income_frame = pd.DataFrame()
        return cashflow_frame, income_frame

    def _extract_quarterly_quality_profile(
        self,
        cashflow_frame: pd.DataFrame,
        income_frame: pd.DataFrame,
    ) -> dict[str, float | str | None]:
        revenue_series = self._extract_statement_series(
            income_frame,
            ("Total Revenue", "TotalRevenue", "Revenue"),
            limit=2,
        )
        operating_income_series = self._extract_statement_series(
            income_frame,
            ("Operating Income", "OperatingIncome", "Operating Income Loss"),
            limit=2,
        )
        operating_cash_flow_series = self._extract_statement_series(
            cashflow_frame,
            ("Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities"),
            limit=2,
        )
        capital_expenditure_series = self._extract_statement_series(
            cashflow_frame,
            ("Capital Expenditure", "CapitalExpenditure"),
            limit=2,
        )

        free_cash_flow_series: list[float] = []
        for operating_cash_flow, capital_expenditure in zip(
            operating_cash_flow_series,
            capital_expenditure_series,
            strict=False,
        ):
            free_cash_flow_series.append(float(operating_cash_flow) - abs(float(capital_expenditure)))

        latest_revenue = revenue_series[0] if revenue_series else None
        latest_fcf = free_cash_flow_series[0] if free_cash_flow_series else None
        free_cash_flow_margin = None
        if latest_fcf is not None and latest_revenue not in {None, 0}:
            free_cash_flow_margin = latest_fcf / float(latest_revenue)

        current_margin = self._compute_margin(
            operating_income_series[0] if len(operating_income_series) >= 1 else None,
            revenue_series[0] if len(revenue_series) >= 1 else None,
        )
        prior_margin = self._compute_margin(
            operating_income_series[1] if len(operating_income_series) >= 2 else None,
            revenue_series[1] if len(revenue_series) >= 2 else None,
        )

        revenue_qoq_change = self._compute_relative_change(
            revenue_series[0] if len(revenue_series) >= 1 else None,
            revenue_series[1] if len(revenue_series) >= 2 else None,
        )
        free_cash_flow_qoq_change = self._compute_relative_change(
            free_cash_flow_series[0] if len(free_cash_flow_series) >= 1 else None,
            free_cash_flow_series[1] if len(free_cash_flow_series) >= 2 else None,
        )
        operating_margin_qoq_change = None
        if current_margin is not None and prior_margin is not None:
            operating_margin_qoq_change = current_margin - prior_margin

        return {
            "free_cash_flow": round(float(latest_fcf), 2) if latest_fcf is not None else None,
            "free_cash_flow_margin": self._as_float(free_cash_flow_margin),
            "revenue_qoq_change": self._as_float(revenue_qoq_change),
            "operating_margin_qoq_change": self._as_float(operating_margin_qoq_change),
            "free_cash_flow_qoq_change": self._as_float(free_cash_flow_qoq_change),
            "revenue_quality_trend": self._classify_qoq_trend(revenue_qoq_change, positive_threshold=0.03, negative_threshold=-0.03),
            "margin_quality_trend": self._classify_qoq_trend(operating_margin_qoq_change, positive_threshold=0.01, negative_threshold=-0.01),
            "fcf_quality_trend": self._classify_qoq_trend(free_cash_flow_qoq_change, positive_threshold=0.08, negative_threshold=-0.08),
        }

    def _extract_free_cash_flow_profile(self, ticker_client: yf.Ticker) -> tuple[float | None, float | None]:
        cashflow_frame, income_frame = self._load_financial_statement_frames(ticker_client)
        profile = self._extract_quarterly_quality_profile(cashflow_frame, income_frame)
        return self._as_float(profile.get("free_cash_flow")), self._as_float(profile.get("free_cash_flow_margin"))

    def _extract_statement_value(self, frame: pd.DataFrame, candidates: Sequence[str]) -> float | None:
        if frame is None or frame.empty:
            return None
        for candidate in candidates:
            if candidate in frame.index:
                series = frame.loc[candidate]
                value = self._first_numeric(series.tolist() if hasattr(series, "tolist") else [series])
                if value is not None:
                    return value
            if candidate in frame.columns:
                series = frame[candidate]
                value = self._first_numeric(series.tolist() if hasattr(series, "tolist") else [series])
                if value is not None:
                    return value
        return None

    def _extract_statement_series(self, frame: pd.DataFrame, candidates: Sequence[str], *, limit: int = 2) -> list[float]:
        if frame is None or frame.empty:
            return []
        for candidate in candidates:
            if candidate in frame.index:
                series = frame.loc[candidate]
                return self._collect_numeric_values(series.tolist() if hasattr(series, "tolist") else [series], limit=limit)
            if candidate in frame.columns:
                series = frame[candidate]
                return self._collect_numeric_values(series.tolist() if hasattr(series, "tolist") else [series], limit=limit)
        return []

    def _first_numeric(self, values: Sequence[Any]) -> float | None:
        for value in values:
            numeric = self._as_float(value)
            if numeric is not None:
                return numeric
        return None

    def _as_int(self, value: Any) -> int | None:
        numeric = self._as_float(value)
        if numeric is None:
            return None
        return int(round(numeric))

    def _collect_numeric_values(self, values: Sequence[Any], *, limit: int) -> list[float]:
        collected: list[float] = []
        for value in values:
            numeric = self._as_float(value)
            if numeric is None:
                continue
            collected.append(numeric)
            if len(collected) >= max(1, limit):
                break
        return collected

    def _compute_relative_change(self, latest: float | None, previous: float | None) -> float | None:
        if latest is None or previous in {None, 0}:
            return None
        return (float(latest) - float(previous)) / abs(float(previous))

    def _compute_margin(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator in {None, 0}:
            return None
        return float(numerator) / float(denominator)

    def _classify_qoq_trend(
        self,
        value: float | None,
        *,
        positive_threshold: float,
        negative_threshold: float,
    ) -> str | None:
        if value is None:
            return None
        if value >= positive_threshold:
            return "improving"
        if value <= negative_threshold:
            return "deteriorating"
        return "stable"

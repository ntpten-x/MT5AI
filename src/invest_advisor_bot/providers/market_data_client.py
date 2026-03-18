from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from loguru import logger

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


class MarketDataClient:
    """Async wrapper around yfinance for quote and OHLCV retrieval."""

    def __init__(
        self,
        asset_universe: Mapping[str, str] | None = None,
        *,
        cache_ttl_seconds: int = 900,
        cache_maxsize: int = 256,
    ) -> None:
        self.asset_universe: dict[str, str] = dict(asset_universe or DEFAULT_ASSET_UNIVERSE)
        self._cache_lock = RLock()
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

    async def get_latest_price(self, ticker: str) -> AssetQuote | None:
        cache_key = ("latest", ticker.upper())
        with self._cache_lock:
            cached = self._latest_price_cache.get(cache_key)
        if cache_key in self._latest_price_cache or cached is not None:
            return cached

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
            cached = self._history_cache.get(cache_key)
        if cache_key in self._history_cache or cached is not None:
            return list(cached or [])

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
            cached = self._snapshot_cache.get(cache_key)
        if cache_key in self._snapshot_cache or cached is not None:
            return dict(cached or {})

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
            cached = self._core_history_cache.get(cache_key)
        if cache_key in self._core_history_cache or cached is not None:
            return {asset: list(bars) for asset, bars in (cached or {}).items()}

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

    def _get_latest_price_sync(self, ticker: str) -> AssetQuote | None:
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

    def _get_history_sync(
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

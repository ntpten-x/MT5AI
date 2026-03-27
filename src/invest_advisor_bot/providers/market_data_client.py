from __future__ import annotations

import asyncio
import importlib
import re
import time
import xml.etree.ElementTree as ET
from io import StringIO
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from html import unescape
from threading import RLock
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import httpx
import yfinance as yf
from cachetools import TTLCache
from loguru import logger

from invest_advisor_bot.runtime_diagnostics import diagnostics
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
class OptionContractSnapshot:
    contract_ticker: str
    underlying_ticker: str
    contract_type: str | None
    expiration_date: str | None
    strike_price: float | None
    bid: float | None
    ask: float | None
    midpoint: float | None
    last_price: float | None
    implied_volatility: float | None
    open_interest: int | None
    volume: int | None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    updated_at: datetime | None = None


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


@dataclass(slots=True, frozen=True)
class AnalystRatingsProfile:
    ticker: str
    consensus_signal: str | None
    buy_count: int | None
    hold_count: int | None
    sell_count: int | None
    target_price: float | None
    upside_pct: float | None
    source: str | None = None


@dataclass(slots=True, frozen=True)
class InsiderTransaction:
    filed_at: datetime | None
    trade_type: str
    shares: float | None
    price_per_share: float | None
    value: float | None
    source_url: str | None = None


@dataclass(slots=True, frozen=True)
class InsiderTransactionSummary:
    ticker: str
    signal: str | None
    net_shares: float | None
    net_value: float | None
    buy_count: int
    sell_count: int
    transaction_count: int
    last_filed_at: datetime | None
    recent_transactions: tuple[InsiderTransaction, ...] = ()


@dataclass(slots=True, frozen=True)
class CorporateActionEvent:
    ticker: str
    action_type: str
    ex_date: datetime | None
    record_date: datetime | None = None
    payable_date: datetime | None = None
    cash_amount: float | None = None
    ratio: float | None = None
    source: str | None = None


@dataclass(slots=True, frozen=True)
class ETFExposureProfile:
    ticker: str
    fund_family: str | None
    category: str | None
    total_assets: float | None
    fund_flow_1m_pct: float | None
    top_holdings: tuple[tuple[str, float | None], ...]
    sector_exposures: tuple[tuple[str, float | None], ...]
    country_exposures: tuple[tuple[str, float | None], ...]
    concentration_score: float | None
    exposure_signal: str | None
    source: str | None = None


@dataclass(slots=True, frozen=True)
class FilingEvent:
    ticker: str
    cik: str | None
    form: str
    filing_date: datetime | None
    report_date: datetime | None
    accession_number: str | None
    primary_document: str | None
    primary_document_url: str | None


@dataclass(slots=True, frozen=True)
class CompanyIntelligence:
    ticker: str
    company_name: str | None
    cik: str | None
    latest_10k_filed_at: datetime | None
    latest_10q_filed_at: datetime | None
    latest_8k_filed_at: datetime | None
    revenue_latest: float | None
    revenue_yoy_pct: float | None
    operating_cash_flow_latest: float | None
    free_cash_flow_latest: float | None
    debt_latest: float | None
    debt_delta_pct: float | None
    share_dilution_yoy_pct: float | None
    one_off_signal: str | None
    guidance_signal: str | None
    insider_signal: str | None
    sentiment_signal: str | None
    earnings_expectation_signal: str | None
    analyst_rating_signal: str | None = None
    analyst_buy_count: int | None = None
    analyst_hold_count: int | None = None
    analyst_sell_count: int | None = None
    analyst_upside_pct: float | None = None
    insider_net_shares: float | None = None
    insider_net_value: float | None = None
    insider_transaction_count: int | None = None
    insider_last_filed_at: datetime | None = None
    corporate_action_signal: str | None = None
    recent_corporate_actions: tuple[CorporateActionEvent, ...] = ()
    filing_highlights: tuple[str, ...] = ()
    recent_filings: tuple[FilingEvent, ...] = ()


@dataclass(slots=True, frozen=True)
class MacroEvent:
    event_key: str
    event_name: str
    category: str
    source: str
    scheduled_at: datetime
    importance: str
    status: str
    source_url: str | None = None
    country: str | None = None
    previous_value: float | None = None
    forecast_value: float | None = None
    actual_value: float | None = None


@dataclass(slots=True, frozen=True)
class MacroSurpriseSignal:
    event_key: str
    event_name: str
    category: str
    source: str
    released_at: datetime | None
    next_event_at: datetime | None
    actual_value: float | None
    expected_value: float | None
    surprise_value: float | None
    surprise_direction: str | None
    surprise_label: str
    market_bias: str | None
    rationale: tuple[str, ...]
    detail_url: str | None = None
    baseline_expected_value: float | None = None
    baseline_surprise_value: float | None = None
    baseline_surprise_label: str | None = None
    consensus_expected_value: float | None = None
    consensus_te_expected_value: float | None = None
    consensus_surprise_value: float | None = None
    consensus_surprise_label: str | None = None


@dataclass(slots=True, frozen=True)
class MacroReactionAssetMove:
    ticker: str
    label: str
    move_5m_pct: float | None
    move_1h_pct: float | None
    expected_direction: str
    confirmed_5m: bool | None
    confirmed_1h: bool | None


@dataclass(slots=True, frozen=True)
class MacroMarketReaction:
    event_key: str
    event_name: str
    released_at: datetime | None
    market_bias: str | None
    confirmation_label: str
    confirmation_score_5m: float | None
    confirmation_score_1h: float | None
    rationale: tuple[str, ...]
    reactions: tuple[MacroReactionAssetMove, ...]


class _OpenBBRestAdapter:
    def __init__(self, *, base_url: str, pat: str, http_client: httpx.Client) -> None:
        self.base_url = base_url.rstrip("/")
        self.pat = pat.strip()
        self.http_client = http_client

    def fetch_quote(self, ticker: str) -> dict[str, Any] | None:
        response = self.http_client.get(
            f"{self.base_url}/api/v1/equity/price/quote",
            params={"symbol": ticker.upper()},
            headers=self._headers(),
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            return dict(payload[0]) if isinstance(payload[0], Mapping) else None
        return dict(payload) if isinstance(payload, Mapping) else None

    def fetch_history(self, ticker: str, *, period: str, interval: str, limit: int | None) -> list[dict[str, Any]]:
        response = self.http_client.get(
            f"{self.base_url}/api/v1/equity/price/historical",
            params={"symbol": ticker.upper(), "period": period, "interval": interval, "limit": limit},
            headers=self._headers(),
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, Mapping):
            results = payload.get("results") or payload.get("data")
            if isinstance(results, list):
                return [dict(item) for item in results if isinstance(item, Mapping)]
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        return []

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "InvestAdvisorBot/0.2"}
        if self.pat:
            headers["Authorization"] = f"Bearer {self.pat}"
        return headers


class _OpenBBSdkAdapter:
    def __init__(self, client: Any) -> None:
        self.client = client

    def fetch_quote(self, ticker: str) -> dict[str, Any] | None:
        price_api = getattr(getattr(getattr(self.client, "equity", None), "price", None), "quote", None)
        if price_api is None:
            return None
        result = price_api(symbol=ticker.upper())
        rows = self._to_rows(result)
        return rows[0] if rows else None

    def fetch_history(self, ticker: str, *, period: str, interval: str, limit: int | None) -> list[dict[str, Any]]:
        price_api = getattr(getattr(getattr(self.client, "equity", None), "price", None), "historical", None)
        if price_api is None:
            return []
        result = price_api(symbol=ticker.upper(), period=period, interval=interval)
        rows = self._to_rows(result)
        return rows[-limit:] if limit is not None and limit > 0 else rows

    @staticmethod
    def _to_rows(result: Any) -> list[dict[str, Any]]:
        if result is None:
            return []
        to_df = getattr(result, "to_df", None)
        if callable(to_df):
            frame = to_df()
            if isinstance(frame, pd.DataFrame):
                if "date" not in frame.columns and frame.index.name:
                    frame = frame.reset_index()
                return frame.to_dict(orient="records")
        if isinstance(result, pd.DataFrame):
            frame = result.reset_index() if result.index.name else result
            return frame.to_dict(orient="records")
        if isinstance(result, list):
            return [dict(item) for item in result if isinstance(item, Mapping)]
        if isinstance(result, Mapping):
            return [dict(result)]
        return []


class MarketDataClient:
    """Async wrapper around yfinance for quote and OHLCV retrieval."""

    def __init__(
        self,
        asset_universe: Mapping[str, str] | None = None,
        *,
        cache_ttl_seconds: int = 900,
        cache_maxsize: int = 256,
        alpha_vantage_api_key: str = "",
        finnhub_api_key: str = "",
        fred_api_key: str = "",
        bls_api_key: str = "",
        eia_api_key: str = "",
        bea_api_key: str = "",
        ecb_api_base_url: str = "https://data-api.ecb.europa.eu/service/data",
        ecb_series_map: str = "",
        imf_api_base_url: str = "",
        imf_series_map: str = "",
        world_bank_api_base_url: str = "https://api.worldbank.org/v2",
        world_bank_countries: Sequence[str] | None = None,
        world_bank_indicator_map: str = "",
        global_macro_calendar_countries: Sequence[str] | None = None,
        global_macro_calendar_importance: int = 2,
        openbb_pat: str = "",
        openbb_base_url: str = "",
        polygon_api_key: str = "",
        polygon_base_url: str = "https://api.polygon.io",
        polygon_options_chain_limit: int = 20,
        cme_fedwatch_api_key: str = "",
        cme_fedwatch_api_url: str = "",
        nasdaq_data_link_api_key: str = "",
        nasdaq_data_link_base_url: str = "https://data.nasdaq.com/api/v3",
        nasdaq_data_link_datasets: Sequence[str] | None = None,
        gdelt_context_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc",
        gdelt_geo_base_url: str = "https://api.gdeltproject.org/api/v2/geo/geo",
        gdelt_query: str = "",
        gdelt_max_records: int = 25,
        tradier_access_token: str = "",
        tradier_base_url: str = "https://api.tradier.com",
        sec_user_agent: str = "InvestAdvisorBot/0.2 support@example.com",
        trading_economics_api_key: str = "",
        provider_order: Sequence[str] | None = None,
        http_timeout_seconds: float = 12.0,
    ) -> None:
        self.asset_universe: dict[str, str] = dict(asset_universe or DEFAULT_ASSET_UNIVERSE)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.alpha_vantage_api_key = alpha_vantage_api_key.strip()
        self.finnhub_api_key = finnhub_api_key.strip()
        self.fred_api_key = fred_api_key.strip()
        self.bls_api_key = bls_api_key.strip()
        self.eia_api_key = eia_api_key.strip()
        self.bea_api_key = bea_api_key.strip()
        self.ecb_api_base_url = ecb_api_base_url.strip().rstrip("/")
        self.ecb_series_map = self._parse_mapping_string(ecb_series_map)
        self.imf_api_base_url = imf_api_base_url.strip().rstrip("/")
        self.imf_series_map = self._parse_mapping_string(imf_series_map)
        self.world_bank_api_base_url = world_bank_api_base_url.strip().rstrip("/") or "https://api.worldbank.org/v2"
        self.world_bank_countries = tuple(
            dict.fromkeys(item.strip().upper() for item in (world_bank_countries or ()) if item and item.strip())
        )
        self.world_bank_indicator_map = self._parse_mapping_string(world_bank_indicator_map)
        self.global_macro_calendar_countries = tuple(
            dict.fromkeys(item.strip() for item in (global_macro_calendar_countries or ()) if item and item.strip())
        )
        self.global_macro_calendar_importance = max(1, int(global_macro_calendar_importance))
        self.openbb_pat = openbb_pat.strip()
        self.openbb_base_url = openbb_base_url.strip().rstrip("/")
        self.polygon_api_key = polygon_api_key.strip()
        self.polygon_base_url = polygon_base_url.strip().rstrip("/") or "https://api.polygon.io"
        self.polygon_options_chain_limit = max(1, int(polygon_options_chain_limit))
        self.cme_fedwatch_api_key = cme_fedwatch_api_key.strip()
        self.cme_fedwatch_api_url = cme_fedwatch_api_url.strip()
        self.nasdaq_data_link_api_key = nasdaq_data_link_api_key.strip()
        self.nasdaq_data_link_base_url = nasdaq_data_link_base_url.strip().rstrip("/") or "https://data.nasdaq.com/api/v3"
        self.nasdaq_data_link_datasets = tuple(
            dict.fromkeys(item.strip() for item in (nasdaq_data_link_datasets or ()) if item and item.strip())
        )
        self._nasdaq_data_link_disabled = False
        self._nasdaq_data_link_warning: str | None = None
        self._trading_economics_calendar_disabled = False
        self._trading_economics_warning: str | None = None
        self.gdelt_context_base_url = gdelt_context_base_url.strip().rstrip("/") or "https://api.gdeltproject.org/api/v2/doc/doc"
        self.gdelt_geo_base_url = gdelt_geo_base_url.strip().rstrip("/") or "https://api.gdeltproject.org/api/v2/geo/geo"
        self.gdelt_query = gdelt_query.strip()
        self.gdelt_max_records = max(1, int(gdelt_max_records))
        self.tradier_access_token = tradier_access_token.strip()
        self.tradier_base_url = tradier_base_url.strip().rstrip("/") or "https://api.tradier.com"
        self.sec_user_agent = sec_user_agent.strip() or "InvestAdvisorBot/0.2 support@example.com"
        self.trading_economics_api_key = trading_economics_api_key.strip()
        normalized_order = tuple(
            item.strip().casefold()
            for item in (provider_order or ("polygon", "alpha_vantage", "yfinance"))
            if item and item.strip()
        )
        if normalized_order:
            self.provider_order = normalized_order
        elif self.polygon_api_key:
            self.provider_order = ("polygon", "alpha_vantage", "yfinance")
        elif self.alpha_vantage_api_key:
            self.provider_order = ("alpha_vantage", "yfinance")
        else:
            self.provider_order = ("yfinance",)
        self.http_timeout_seconds = max(2.0, float(http_timeout_seconds))
        self._alpha_vantage_disabled = False
        self._alpha_vantage_warning: str | None = None
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
        self._macro_intelligence_cache: TTLCache[str, dict[str, Any]] = TTLCache(
            maxsize=8,
            ttl=cache_ttl_seconds,
        )
        self._global_event_cache: TTLCache[str, dict[str, Any]] = TTLCache(
            maxsize=8,
            ttl=max(300, min(cache_ttl_seconds, 3600)),
        )
        self._macro_event_calendar_cache: TTLCache[str, tuple[MacroEvent, ...]] = TTLCache(
            maxsize=8,
            ttl=max(cache_ttl_seconds, 3600),
        )
        self._macro_surprise_cache: TTLCache[str, tuple[MacroSurpriseSignal, ...]] = TTLCache(
            maxsize=8,
            ttl=max(cache_ttl_seconds, 3600),
        )
        self._macro_market_reaction_cache: TTLCache[str, tuple[MacroMarketReaction, ...]] = TTLCache(
            maxsize=8,
            ttl=max(cache_ttl_seconds, 900),
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
        self._options_chain_cache: TTLCache[tuple[str, str | None, str | None, int], list[OptionContractSnapshot]] = TTLCache(
            maxsize=64,
            ttl=max(60, min(cache_ttl_seconds, 300)),
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
        self._company_intelligence_cache: TTLCache[str, CompanyIntelligence | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=max(cache_ttl_seconds, 3600),
        )
        self._analyst_rating_cache: TTLCache[str, AnalystRatingsProfile | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=max(cache_ttl_seconds, 1800),
        )
        self._etf_exposure_cache: TTLCache[str, ETFExposureProfile | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=max(cache_ttl_seconds, 1800),
        )
        self._insider_summary_cache: TTLCache[str, InsiderTransactionSummary | None] = TTLCache(
            maxsize=cache_maxsize,
            ttl=max(cache_ttl_seconds, 3600),
        )
        self._corporate_actions_cache: TTLCache[str, tuple[CorporateActionEvent, ...]] = TTLCache(
            maxsize=cache_maxsize,
            ttl=max(cache_ttl_seconds, 3600),
        )
        self._sec_ticker_cache: TTLCache[str, dict[str, dict[str, str]]] = TTLCache(
            maxsize=1,
            ttl=max(cache_ttl_seconds, 6 * 3600),
        )
        self._fred_release_cache: TTLCache[str, int | None] = TTLCache(
            maxsize=16,
            ttl=max(cache_ttl_seconds, 6 * 3600),
        )
        self._document_text_cache: TTLCache[str, str | None] = TTLCache(
            maxsize=128,
            ttl=max(cache_ttl_seconds, 6 * 3600),
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

    def status(self) -> dict[str, Any]:
        with self._cache_lock:
            cache_state = {
                "latest_price_entries": len(self._latest_price_cache),
                "history_entries": len(self._history_cache),
                "snapshot_entries": len(self._snapshot_cache),
                "macro_context_entries": len(self._macro_context_cache),
                "macro_intelligence_entries": len(self._macro_intelligence_cache),
                "global_event_entries": len(self._global_event_cache),
                "fundamentals_entries": len(self._fundamentals_cache),
                "company_intelligence_entries": len(self._company_intelligence_cache),
                "analyst_rating_entries": len(self._analyst_rating_cache),
                "etf_exposure_entries": len(self._etf_exposure_cache),
                "insider_summary_entries": len(self._insider_summary_cache),
                "corporate_actions_entries": len(self._corporate_actions_cache),
                "options_chain_entries": len(self._options_chain_cache),
            }
        configured_sources = {
            "polygon": bool(self.polygon_api_key),
            "alpha_vantage": bool(self.alpha_vantage_api_key and not self._alpha_vantage_disabled),
            "finnhub": bool(self.finnhub_api_key),
            "fred": bool(self.fred_api_key),
            "bls": bool(self.bls_api_key),
            "eia": bool(self.eia_api_key),
            "bea": bool(self.bea_api_key),
            "ecb": bool(self.ecb_series_map),
            "imf": bool(self.imf_series_map and self.imf_api_base_url),
            "world_bank": bool(self.world_bank_indicator_map and self.world_bank_countries),
            "global_macro_calendar": bool(self.global_macro_calendar_countries),
            "sec": bool(self.sec_user_agent),
            "cftc": True,
            "openbb": bool(self.openbb_pat or self.openbb_base_url),
            "trading_economics": bool(self.trading_economics_api_key and not self._trading_economics_calendar_disabled),
            "cme_fedwatch": bool(self.cme_fedwatch_api_url),
            "nasdaq_data_link": bool(
                self.nasdaq_data_link_api_key
                and self.nasdaq_data_link_datasets
                and not self._nasdaq_data_link_disabled
            ),
            "gdelt": bool(self.gdelt_query),
            "tradier": bool(self.tradier_access_token),
            "yfinance": True,
        }
        return {
            "available": True,
            "provider_order": list(self.provider_order),
            "configured_sources": configured_sources,
            "asset_universe_size": len(self.asset_universe),
            "http_timeout_seconds": self.http_timeout_seconds,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "polygon_options_chain_limit": self.polygon_options_chain_limit,
            "alpha_vantage_disabled": self._alpha_vantage_disabled,
            "alpha_vantage_warning": self._alpha_vantage_warning,
            "nasdaq_data_link_datasets": list(self.nasdaq_data_link_datasets),
            "nasdaq_data_link_disabled": self._nasdaq_data_link_disabled,
            "nasdaq_data_link_warning": self._nasdaq_data_link_warning,
            "trading_economics_calendar_disabled": self._trading_economics_calendar_disabled,
            "trading_economics_warning": self._trading_economics_warning,
            "gdelt_query": self.gdelt_query or None,
            "cache_state": cache_state,
        }

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

    async def get_option_chain_snapshot(
        self,
        ticker: str,
        *,
        expiration_date: str | None = None,
        contract_type: str | None = None,
        limit: int | None = None,
    ) -> list[OptionContractSnapshot]:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return []
        effective_limit = max(1, int(limit or self.polygon_options_chain_limit))
        cache_key = (normalized_ticker, expiration_date, contract_type, effective_limit)
        with self._cache_lock:
            if cache_key in self._options_chain_cache:
                return list(self._options_chain_cache[cache_key] or [])
        try:
            result = await asyncio.to_thread(
                self._get_option_chain_sync,
                normalized_ticker,
                expiration_date,
                contract_type,
                effective_limit,
            )
        except Exception as exc:
            logger.warning("Failed to fetch option chain for {}: {}", normalized_ticker, exc)
            return []
        with self._cache_lock:
            self._options_chain_cache[cache_key] = list(result)
        return result

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

    async def get_analyst_rating_profile(self, ticker: str) -> AnalystRatingsProfile | None:
        cache_key = ticker.upper()
        with self._cache_lock:
            if cache_key in self._analyst_rating_cache:
                return self._analyst_rating_cache[cache_key]
        try:
            result = await asyncio.to_thread(self._get_analyst_rating_profile_sync, ticker)
        except Exception as exc:
            logger.exception("Failed to fetch analyst rating profile for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._analyst_rating_cache[cache_key] = result
        return result

    async def get_analyst_rating_profiles(self, tickers: Sequence[str]) -> dict[str, AnalystRatingsProfile | None]:
        tasks = {ticker.upper(): self.get_analyst_rating_profile(ticker) for ticker in tickers if ticker.strip()}
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, AnalystRatingsProfile | None] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            payload[ticker] = None if isinstance(result, Exception) else result
        return payload

    async def get_insider_transaction_summary(self, ticker: str) -> InsiderTransactionSummary | None:
        cache_key = ticker.upper()
        with self._cache_lock:
            if cache_key in self._insider_summary_cache:
                return self._insider_summary_cache[cache_key]
        try:
            result = await asyncio.to_thread(self._get_insider_transaction_summary_sync, ticker)
        except Exception as exc:
            logger.exception("Failed to fetch insider transaction summary for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._insider_summary_cache[cache_key] = result
        return result

    async def get_insider_transaction_summaries(self, tickers: Sequence[str]) -> dict[str, InsiderTransactionSummary | None]:
        tasks = {ticker.upper(): self.get_insider_transaction_summary(ticker) for ticker in tickers if ticker.strip()}
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, InsiderTransactionSummary | None] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            payload[ticker] = None if isinstance(result, Exception) else result
        return payload

    async def get_corporate_actions(self, ticker: str, *, lookback_days: int = 365) -> list[CorporateActionEvent]:
        cache_key = f"{ticker.upper()}::{max(30, int(lookback_days))}"
        with self._cache_lock:
            if cache_key in self._corporate_actions_cache:
                return list(self._corporate_actions_cache[cache_key] or ())
        try:
            result = await asyncio.to_thread(self._get_corporate_actions_sync, ticker, lookback_days)
        except Exception as exc:
            logger.exception("Failed to fetch corporate actions for {}: {}", ticker, exc)
            return []
        with self._cache_lock:
            self._corporate_actions_cache[cache_key] = tuple(result)
        return result

    async def get_corporate_actions_batch(
        self,
        tickers: Sequence[str],
        *,
        lookback_days: int = 365,
    ) -> dict[str, list[CorporateActionEvent]]:
        tasks = {
            ticker.upper(): self.get_corporate_actions(ticker, lookback_days=lookback_days)
            for ticker in tickers
            if ticker.strip()
        }
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, list[CorporateActionEvent]] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            payload[ticker] = [] if isinstance(result, Exception) else list(result)
        return payload

    async def get_etf_exposure_profile(self, ticker: str) -> ETFExposureProfile | None:
        cache_key = ticker.upper()
        with self._cache_lock:
            if cache_key in self._etf_exposure_cache:
                return self._etf_exposure_cache[cache_key]
        try:
            result = await asyncio.to_thread(self._get_etf_exposure_profile_sync, ticker)
        except Exception as exc:
            logger.exception("Failed to fetch ETF exposure profile for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._etf_exposure_cache[cache_key] = result
        return result

    async def get_etf_exposure_profiles(self, tickers: Sequence[str]) -> dict[str, ETFExposureProfile | None]:
        tasks = {ticker.upper(): self.get_etf_exposure_profile(ticker) for ticker in tickers if ticker.strip()}
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, ETFExposureProfile | None] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            payload[ticker] = None if isinstance(result, Exception) else result
        return payload

    async def get_macro_intelligence(self) -> dict[str, Any]:
        cache_key = "macro_intelligence"
        with self._cache_lock:
            if cache_key in self._macro_intelligence_cache:
                return dict(self._macro_intelligence_cache[cache_key])
        result = await asyncio.to_thread(self._get_macro_intelligence_sync)
        with self._cache_lock:
            self._macro_intelligence_cache[cache_key] = dict(result)
        return result

    async def get_macro_event_calendar(self, *, days_ahead: int = 30) -> list[MacroEvent]:
        normalized_days = max(3, int(days_ahead))
        cache_key = f"macro_event_calendar:{normalized_days}"
        with self._cache_lock:
            if cache_key in self._macro_event_calendar_cache:
                return list(self._macro_event_calendar_cache[cache_key])
        result = await asyncio.to_thread(self._get_macro_event_calendar_sync, normalized_days)
        with self._cache_lock:
            self._macro_event_calendar_cache[cache_key] = tuple(result)
        return result

    async def get_macro_surprise_signals(self) -> list[MacroSurpriseSignal]:
        cache_key = "macro_surprise_signals"
        with self._cache_lock:
            if cache_key in self._macro_surprise_cache:
                return list(self._macro_surprise_cache[cache_key])
        result = await asyncio.to_thread(self._get_macro_surprise_signals_sync)
        with self._cache_lock:
            self._macro_surprise_cache[cache_key] = tuple(result)
        return result

    async def get_macro_market_reactions(self) -> list[MacroMarketReaction]:
        cache_key = "macro_market_reactions"
        with self._cache_lock:
            if cache_key in self._macro_market_reaction_cache:
                return list(self._macro_market_reaction_cache[cache_key])
        result = await asyncio.to_thread(self._get_macro_market_reactions_sync)
        with self._cache_lock:
            self._macro_market_reaction_cache[cache_key] = tuple(result)
        return result

    async def get_company_intelligence(
        self,
        ticker: str,
        *,
        company_name: str | None = None,
    ) -> CompanyIntelligence | None:
        cache_key = ticker.upper()
        with self._cache_lock:
            if cache_key in self._company_intelligence_cache:
                return self._company_intelligence_cache[cache_key]
        try:
            result = await asyncio.to_thread(self._get_company_intelligence_sync, ticker, company_name)
        except Exception as exc:
            logger.exception("Failed to fetch company intelligence for {}: {}", ticker, exc)
            return None
        with self._cache_lock:
            self._company_intelligence_cache[cache_key] = result
        return result

    async def get_company_intelligence_batch(
        self,
        tickers: Sequence[str],
        *,
        company_names: Mapping[str, str] | None = None,
    ) -> dict[str, CompanyIntelligence | None]:
        tasks = {
            ticker.upper(): self.get_company_intelligence(ticker, company_name=(company_names or {}).get(ticker.upper()))
            for ticker in tickers
            if ticker.strip()
        }
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, CompanyIntelligence | None] = {}
        for ticker, result in zip(tasks.keys(), results, strict=False):
            payload[ticker] = None if isinstance(result, Exception) else result
        return payload

    def _get_latest_price_sync(self, ticker: str) -> AssetQuote | None:
        providers = self._effective_provider_order()
        for provider_name in providers:
            started_at = time.perf_counter()
            try:
                if provider_name == "polygon":
                    quote = self._get_latest_price_polygon_sync(ticker)
                elif provider_name == "openbb":
                    quote = self._get_latest_price_openbb_sync(ticker)
                elif provider_name == "alpha_vantage":
                    quote = self._get_latest_price_alpha_vantage_sync(ticker)
                elif provider_name == "yfinance":
                    quote = self._get_latest_price_yfinance_sync(ticker)
                else:
                    continue
            except Exception as exc:
                diagnostics.record_provider_latency(
                    service="market_data_client",
                    provider=provider_name,
                    operation="latest_price",
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    success=False,
                )
                logger.warning("Market data provider {} latest price failed for {}: {}", provider_name, ticker, exc)
                continue
            diagnostics.record_provider_latency(
                service="market_data_client",
                provider=provider_name,
                operation="latest_price",
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                success=quote is not None,
            )
            if quote is not None:
                return quote
        return None

    def _get_latest_price_polygon_sync(self, ticker: str) -> AssetQuote | None:
        normalized = ticker.strip().upper()
        if not normalized or not self.polygon_api_key:
            return None
        payload = self._fetch_polygon_json(f"/v2/snapshot/locale/us/markets/stocks/tickers/{normalized}")
        if not isinstance(payload, Mapping):
            return None
        ticker_payload = payload.get("ticker")
        if not isinstance(ticker_payload, Mapping):
            return None
        day_payload = ticker_payload.get("day") if isinstance(ticker_payload.get("day"), Mapping) else {}
        minute_payload = ticker_payload.get("min") if isinstance(ticker_payload.get("min"), Mapping) else {}
        prev_day_payload = ticker_payload.get("prevDay") if isinstance(ticker_payload.get("prevDay"), Mapping) else {}
        last_trade_payload = ticker_payload.get("lastTrade") if isinstance(ticker_payload.get("lastTrade"), Mapping) else {}
        price = (
            self._as_float(minute_payload.get("c"))
            or self._as_float(last_trade_payload.get("p"))
            or self._as_float(day_payload.get("c"))
            or self._as_float(prev_day_payload.get("c"))
        )
        if price is None or price <= 0:
            return None
        return AssetQuote(
            ticker=normalized,
            name=str(ticker_payload.get("name") or normalized).strip(),
            currency="USD",
            exchange=self._as_optional_str(ticker_payload.get("primary_exchange")),
            price=float(price),
            previous_close=self._as_float(prev_day_payload.get("c")),
            open_price=self._as_float(day_payload.get("o")),
            day_high=self._as_float(day_payload.get("h")),
            day_low=self._as_float(day_payload.get("l")),
            volume=self._as_int(day_payload.get("v")),
            timestamp=self._parse_polygon_timestamp(last_trade_payload.get("t") or ticker_payload.get("updated")),
        )

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
            started_at = time.perf_counter()
            try:
                if provider_name == "polygon":
                    bars = self._get_history_polygon_sync(ticker, period=period, interval=interval, limit=limit)
                elif provider_name == "openbb":
                    bars = self._get_history_openbb_sync(ticker, period=period, interval=interval, limit=limit)
                elif provider_name == "alpha_vantage":
                    bars = self._get_history_alpha_vantage_sync(ticker, period=period, interval=interval, limit=limit)
                elif provider_name == "yfinance":
                    bars = self._get_history_yfinance_sync(ticker, period=period, interval=interval, limit=limit)
                else:
                    continue
            except Exception as exc:
                diagnostics.record_provider_latency(
                    service="market_data_client",
                    provider=provider_name,
                    operation="history",
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    success=False,
                )
                logger.warning(
                    "Market data provider {} history failed for {} (period={}, interval={}): {}",
                    provider_name,
                    ticker,
                    period,
                    interval,
                    exc,
                )
                continue
            diagnostics.record_provider_latency(
                service="market_data_client",
                provider=provider_name,
                operation="history",
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                success=bool(bars),
            )
            if bars:
                return bars
        return []

    def _get_history_polygon_sync(
        self,
        ticker: str,
        *,
        period: str,
        interval: str,
        limit: int | None,
    ) -> list[OhlcvBar]:
        normalized = ticker.strip().upper()
        if not normalized or not self.polygon_api_key:
            return []
        translated = self._translate_polygon_history_request(period=period, interval=interval)
        if translated is None:
            return []
        multiplier, timespan, start_date, end_date = translated
        payload = self._fetch_polygon_json(
            f"/v2/aggs/ticker/{normalized}/range/{multiplier}/{timespan}/{start_date}/{end_date}",
            params={"adjusted": "true", "sort": "asc", "limit": max(1, int(limit or 5000))},
        )
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list):
            return []
        bars: list[OhlcvBar] = []
        for item in results:
            if not isinstance(item, Mapping):
                continue
            timestamp = self._parse_polygon_timestamp(item.get("t"))
            open_price = self._as_float(item.get("o"))
            high = self._as_float(item.get("h"))
            low = self._as_float(item.get("l"))
            close = self._as_float(item.get("c"))
            volume = self._as_int(item.get("v"))
            if None in {timestamp, open_price, high, low, close}:
                continue
            bars.append(
                OhlcvBar(
                    ticker=normalized,
                    timestamp=timestamp,
                    open=float(open_price),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=int(volume or 0),
                )
            )
        return bars[-limit:] if limit is not None and limit > 0 else bars

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

    def _get_latest_price_openbb_sync(self, ticker: str) -> AssetQuote | None:
        provider = self._load_openbb_provider()
        if provider is None:
            return None
        try:
            payload = provider.fetch_quote(ticker)
        except Exception as exc:
            logger.warning("OpenBB quote request failed for {}: {}", ticker, exc)
            return None
        if not isinstance(payload, Mapping):
            return None
        price = self._as_float(payload.get("price") or payload.get("last_price") or payload.get("close"))
        if price is None:
            return None
        return AssetQuote(
            ticker=ticker.upper(),
            name=str(payload.get("name") or payload.get("symbol") or ticker.upper()),
            currency=self._as_optional_str(payload.get("currency")),
            exchange=self._as_optional_str(payload.get("exchange")),
            price=price,
            previous_close=self._as_float(payload.get("previous_close")),
            open_price=self._as_float(payload.get("open")),
            day_high=self._as_float(payload.get("high")),
            day_low=self._as_float(payload.get("low")),
            volume=self._as_int(payload.get("volume")),
            timestamp=self._parse_datetime(payload.get("timestamp")),
        )

    def _get_history_openbb_sync(
        self,
        ticker: str,
        *,
        period: str,
        interval: str,
        limit: int | None,
    ) -> list[OhlcvBar]:
        provider = self._load_openbb_provider()
        if provider is None:
            return []
        try:
            rows = provider.fetch_history(ticker, period=period, interval=interval, limit=limit)
        except Exception as exc:
            logger.warning("OpenBB history request failed for {}: {}", ticker, exc)
            return []
        bars: list[OhlcvBar] = []
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            timestamp = self._parse_datetime(item.get("timestamp") or item.get("date"))
            open_price = self._as_float(item.get("open"))
            high = self._as_float(item.get("high"))
            low = self._as_float(item.get("low"))
            close = self._as_float(item.get("close"))
            if timestamp is None or None in {open_price, high, low, close}:
                continue
            bars.append(
                OhlcvBar(
                    ticker=ticker.upper(),
                    timestamp=timestamp,
                    open=float(open_price),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=int(self._as_int(item.get("volume")) or 0),
                )
            )
        return bars[-limit:] if limit is not None and limit > 0 else bars

    def _effective_provider_order(self) -> tuple[str, ...]:
        providers: list[str] = []
        for provider_name in self.provider_order:
            if provider_name == "polygon" and not self.polygon_api_key:
                continue
            if provider_name == "openbb" and not self._openbb_available():
                continue
            if provider_name == "alpha_vantage" and not self.alpha_vantage_api_key:
                continue
            if provider_name in {"polygon", "openbb", "alpha_vantage", "yfinance"} and provider_name not in providers:
                providers.append(provider_name)
        if "yfinance" not in providers:
            providers.append("yfinance")
        return tuple(providers)

    def _fetch_polygon_json(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.polygon_api_key:
            return None
        request_params = {"apiKey": self.polygon_api_key}
        if params:
            request_params.update({str(key): value for key, value in params.items() if value is not None})
        try:
            response = self._get_http_client().get(
                f"{self.polygon_base_url}{path}",
                params=request_params,
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Polygon request failed for {}: {}", path, exc)
            return None
        if not isinstance(payload, Mapping):
            return None
        return dict(payload)

    def _fetch_alpha_vantage_json(
        self,
        *,
        function: str,
        symbol: str,
        extra_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.alpha_vantage_api_key or self._alpha_vantage_disabled:
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
            if self._handle_alpha_vantage_non_data_response(note):
                return None
            logger.warning("Alpha Vantage returned non-data response for {} ({}): {}", symbol, function, note)
            return None
        return dict(payload)

    def _fetch_alpha_vantage_csv(
        self,
        *,
        function: str,
        symbol: str,
        extra_params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not self.alpha_vantage_api_key or self._alpha_vantage_disabled:
            return pd.DataFrame()
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
            if self._handle_alpha_vantage_non_data_response(response.text):
                return pd.DataFrame()
            return pd.read_csv(StringIO(response.text))
        except Exception as exc:
            logger.warning("Alpha Vantage CSV request failed for {} ({}): {}", symbol, function, exc)
            return pd.DataFrame()

    def _handle_alpha_vantage_non_data_response(self, note: object) -> bool:
        text = str(note or "").strip()
        if not text:
            return False
        normalized = text.casefold()
        if "25 requests per day" in normalized or "daily rate limit" in normalized:
            if not self._alpha_vantage_disabled:
                self._alpha_vantage_disabled = True
                self._alpha_vantage_warning = "alpha vantage disabled for current runtime after free daily-limit response"
                logger.warning("Alpha Vantage disabled after free daily-limit response")
            return True
        if "rate limit" in normalized or "1 request per second" in normalized:
            self._alpha_vantage_warning = "alpha vantage rate limited on free tier"
            return True
        return False

    def _fetch_finnhub_json(self, endpoint: str, *, symbol: str) -> Any:
        if not self.finnhub_api_key:
            return None
        try:
            response = self._get_http_client().get(
                f"https://finnhub.io/api/v1/{endpoint.lstrip('/')}",
                params={"symbol": symbol, "token": self.finnhub_api_key},
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Finnhub request failed for {} ({}): {}", symbol, endpoint, exc)
            return None
        return payload

    def _get_option_chain_polygon_sync(
        self,
        ticker: str,
        expiration_date: str | None,
        contract_type: str | None,
        limit: int,
    ) -> list[OptionContractSnapshot]:
        if not self.polygon_api_key:
            return []
        params: dict[str, Any] = {
            "limit": max(1, int(limit)),
            "sort": "expiration_date",
        }
        if expiration_date:
            params["expiration_date"] = expiration_date
        if contract_type:
            params["contract_type"] = str(contract_type).strip().lower()
        payload = self._fetch_polygon_json(f"/v3/snapshot/options/{ticker.upper()}", params=params)
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list):
            return []
        snapshots: list[OptionContractSnapshot] = []
        for item in results[:limit]:
            if not isinstance(item, Mapping):
                continue
            details = item.get("details") if isinstance(item.get("details"), Mapping) else {}
            day = item.get("day") if isinstance(item.get("day"), Mapping) else {}
            quote = item.get("last_quote") if isinstance(item.get("last_quote"), Mapping) else {}
            greeks = item.get("greeks") if isinstance(item.get("greeks"), Mapping) else {}
            bid = self._as_float(quote.get("bid"))
            ask = self._as_float(quote.get("ask"))
            midpoint = None if bid is None or ask is None else round((bid + ask) / 2.0, 4)
            snapshots.append(
                OptionContractSnapshot(
                    contract_ticker=str(details.get("ticker") or item.get("ticker") or "").strip(),
                    underlying_ticker=ticker.upper(),
                    contract_type=self._as_optional_str(details.get("contract_type")),
                    expiration_date=self._as_optional_str(details.get("expiration_date")),
                    strike_price=self._as_float(details.get("strike_price")),
                    bid=bid,
                    ask=ask,
                    midpoint=midpoint,
                    last_price=self._as_float(day.get("close")),
                    implied_volatility=self._as_float(item.get("implied_volatility")),
                    open_interest=self._as_int(item.get("open_interest")),
                    volume=self._as_int(day.get("volume")),
                    delta=self._as_float(greeks.get("delta")),
                    gamma=self._as_float(greeks.get("gamma")),
                    theta=self._as_float(greeks.get("theta")),
                    vega=self._as_float(greeks.get("vega")),
                    updated_at=self._parse_polygon_timestamp(item.get("last_updated")),
                )
            )
        return [item for item in snapshots if item.contract_ticker]

    def _get_option_chain_tradier_sync(
        self,
        ticker: str,
        expiration_date: str | None,
        contract_type: str | None,
        limit: int,
    ) -> list[OptionContractSnapshot]:
        if not self.tradier_access_token:
            return []
        normalized = ticker.strip().upper()
        if not normalized:
            return []
        params = {"symbol": normalized, "greeks": "true"}
        if expiration_date:
            params["expiration"] = expiration_date
        try:
            response = self._get_http_client().get(
                f"{self.tradier_base_url}/v1/markets/options/chains",
                params=params,
                headers={
                    "Authorization": f"Bearer {self.tradier_access_token}",
                    "Accept": "application/json",
                    "User-Agent": "InvestAdvisorBot/0.2",
                },
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Tradier option chain request failed for {}: {}", normalized, exc)
            return []
        options_root = payload.get("options") if isinstance(payload, Mapping) and isinstance(payload.get("options"), Mapping) else {}
        option_items = options_root.get("option")
        if isinstance(option_items, Mapping):
            option_items = [option_items]
        if not isinstance(option_items, list):
            return []
        snapshots: list[OptionContractSnapshot] = []
        normalized_contract_type = str(contract_type or "").strip().lower()
        for item in option_items:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("option_type") or "").strip().lower()
            if normalized_contract_type and item_type and normalized_contract_type != item_type:
                continue
            bid = self._as_float(item.get("bid"))
            ask = self._as_float(item.get("ask"))
            midpoint = None if bid is None or ask is None else round((bid + ask) / 2.0, 4)
            snapshots.append(
                OptionContractSnapshot(
                    contract_ticker=str(item.get("symbol") or "").strip(),
                    underlying_ticker=normalized,
                    contract_type=item_type or None,
                    expiration_date=self._as_optional_str(item.get("expiration_date")),
                    strike_price=self._as_float(item.get("strike")),
                    bid=bid,
                    ask=ask,
                    midpoint=midpoint,
                    last_price=self._as_float(item.get("last")),
                    implied_volatility=self._as_float(item.get("greeks", {}).get("mid_iv"))
                    if isinstance(item.get("greeks"), Mapping)
                    else self._as_float(item.get("iv")),
                    open_interest=self._as_int(item.get("open_interest")),
                    volume=self._as_int(item.get("volume")),
                    delta=self._as_float(item.get("greeks", {}).get("delta"))
                    if isinstance(item.get("greeks"), Mapping)
                    else None,
                    gamma=self._as_float(item.get("greeks", {}).get("gamma"))
                    if isinstance(item.get("greeks"), Mapping)
                    else None,
                    theta=self._as_float(item.get("greeks", {}).get("theta"))
                    if isinstance(item.get("greeks"), Mapping)
                    else None,
                    vega=self._as_float(item.get("greeks", {}).get("vega"))
                    if isinstance(item.get("greeks"), Mapping)
                    else None,
                    updated_at=self._parse_datetime(item.get("trade_date")),
                )
            )
            if len(snapshots) >= limit:
                break
        return [item for item in snapshots if item.contract_ticker]

    def _get_option_chain_sync(
        self,
        ticker: str,
        expiration_date: str | None,
        contract_type: str | None,
        limit: int,
    ) -> list[OptionContractSnapshot]:
        polygon_items = self._get_option_chain_polygon_sync(ticker, expiration_date, contract_type, limit)
        if polygon_items:
            return polygon_items
        return self._get_option_chain_tradier_sync(ticker, expiration_date, contract_type, limit)

    def _get_http_client(self) -> httpx.Client:
        with self._http_client_lock:
            client = self._http_client
            if client is None:
                client = httpx.Client(timeout=self.http_timeout_seconds)
                self._http_client = client
            return client

    def _openbb_available(self) -> bool:
        return bool(self.openbb_base_url or self.openbb_pat or self._load_openbb_provider() is not None)

    def _load_openbb_provider(self) -> Any | None:
        if self.openbb_base_url:
            return _OpenBBRestAdapter(base_url=self.openbb_base_url, pat=self.openbb_pat, http_client=self._get_http_client())
        try:
            module = importlib.import_module("openbb")
        except Exception:
            return None
        client = getattr(module, "obb", None)
        if client is None:
            return None
        return _OpenBBSdkAdapter(client)

    @staticmethod
    def _parse_mapping_string(value: str) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for item in str(value or "").split(","):
            raw = item.strip()
            if not raw or "=" not in raw:
                continue
            key, raw_value = raw.split("=", 1)
            normalized_key = key.strip()
            normalized_value = raw_value.strip()
            if normalized_key and normalized_value:
                mapping[normalized_key] = normalized_value
        return mapping

    @staticmethod
    def _normalize_alpha_vantage_symbol(ticker: str) -> str | None:
        normalized = ticker.strip().upper()
        if not normalized or "^" in normalized or "=" in normalized:
            return None
        return normalized.replace("-", ".")

    def _translate_polygon_history_request(self, *, period: str, interval: str) -> tuple[int, str, str, str] | None:
        interval_name = str(interval or "1d").strip().casefold()
        period_days = self._period_to_days(period) or 365
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=max(2, period_days + 5))
        mapping = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "60m": (60, "minute"),
            "1h": (1, "hour"),
            "1d": (1, "day"),
            "1wk": (1, "week"),
            "1mo": (1, "month"),
            "3mo": (3, "month"),
        }
        translated = mapping.get(interval_name)
        if translated is None:
            return None
        multiplier, timespan = translated
        return multiplier, timespan, start_date.isoformat(), end_date.isoformat()

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
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "null":
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = pd.Timestamp(text).to_pydatetime()
            except Exception:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _parse_polygon_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            raw = float(value)
            if raw > 1_000_000_000_000_000:
                raw /= 1_000_000_000.0
            elif raw > 1_000_000_000_000:
                raw /= 1000.0
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        return MarketDataClient._parse_datetime(value)

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
        """Fetch a normalized macro context while preserving legacy core keys."""
        cache_key = "macro_context"
        with self._cache_lock:
            if cache_key in self._macro_context_cache:
                return dict(self._macro_context_cache[cache_key])
        result = await asyncio.to_thread(self._get_macro_context_sync)
        with self._cache_lock:
            self._macro_context_cache[cache_key] = dict(result)
        return result

    def _get_macro_context_sync(self) -> dict[str, float | None]:
        intelligence = self._get_macro_intelligence_sync()
        metrics = intelligence.get("metrics")
        if not isinstance(metrics, Mapping):
            return {"vix": None, "tnx": None, "cpi_yoy": None}
        payload: dict[str, float | None] = {}
        for key, value in metrics.items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            payload[normalized_key] = self._as_float(value)
        payload.setdefault("vix", None)
        payload.setdefault("tnx", None)
        payload.setdefault("cpi_yoy", None)
        return payload

    def _get_macro_intelligence_sync(self) -> dict[str, Any]:
        metrics: dict[str, float | None] = {
            "vix": None,
            "tnx": None,
            "cpi_yoy": None,
            "core_cpi_yoy": None,
            "pce_yoy": None,
            "core_pce_yoy": None,
            "ppi_yoy": None,
            "gdp_qoq_annualized": None,
            "personal_income_mom": None,
            "personal_spending_mom": None,
            "yield_spread_10y_2y": None,
            "high_yield_spread": None,
            "mortgage_30y": None,
            "financial_conditions_index": None,
            "unemployment_rate": None,
            "payrolls_mom_k": None,
            "payrolls_revision_k": None,
            "alfred_payroll_revision_k": None,
            "alfred_cpi_revision_pct": None,
            "avg_interest_rate_pct": None,
            "operating_cash_balance_b": None,
            "public_debt_total_t": None,
            "wti_usd": None,
            "brent_usd": None,
            "gasoline_usd_gal": None,
            "natgas_usd_mmbtu": None,
            "cftc_equity_net_pct_oi": None,
            "cftc_ust10y_net_pct_oi": None,
            "cftc_usd_net_pct_oi": None,
            "cftc_gold_net_pct_oi": None,
            "finra_spy_short_volume_ratio": None,
            "finra_qqq_short_volume_ratio": None,
            "fedwatch_next_meeting_cut_prob_pct": None,
            "fedwatch_next_meeting_hold_prob_pct": None,
            "fedwatch_next_meeting_hike_prob_pct": None,
            "fedwatch_easing_12m_prob_pct": None,
            "fedwatch_hike_12m_prob_pct": None,
            "fedwatch_target_rate_mid_pct": None,
        }
        highlights: list[str] = []
        signals: list[str] = []
        sources_used: list[str] = []
        revisions: dict[str, Any] = {}
        positioning: dict[str, Any] = {}
        qualitative: dict[str, Any] = {}
        short_flow: dict[str, Any] = {}
        fedwatch: dict[str, Any] = {}
        structured_macro: dict[str, Any] = {}
        global_event: dict[str, Any] = {}
        ex_us_macro: dict[str, Any] = {}

        try:
            metrics["vix"] = self._as_float(yf.Ticker("^VIX").fast_info.get("lastPrice"))
        except Exception as exc:
            logger.warning("Failed to fetch VIX: {}", exc)
        try:
            metrics["tnx"] = self._as_float(yf.Ticker("^TNX").fast_info.get("lastPrice"))
        except Exception as exc:
            logger.warning("Failed to fetch TNX: {}", exc)

        fred_series = {
            "cpi": self._fetch_fred_series_frame("CPIAUCSL"),
            "core_cpi": self._fetch_fred_series_frame("CPILFESL"),
            "yield_spread_10y_2y": self._fetch_fred_series_frame("T10Y2Y"),
            "high_yield_spread": self._fetch_fred_series_frame("BAMLH0A0HYM2"),
            "mortgage_30y": self._fetch_fred_series_frame("MORTGAGE30US"),
            "financial_conditions_index": self._fetch_fred_series_frame("NFCI"),
        }
        metrics["cpi_yoy"] = self._compute_yoy_from_frame(fred_series["cpi"])
        metrics["core_cpi_yoy"] = self._compute_yoy_from_frame(fred_series["core_cpi"])
        metrics["yield_spread_10y_2y"] = self._latest_frame_value(fred_series["yield_spread_10y_2y"])
        metrics["high_yield_spread"] = self._latest_frame_value(fred_series["high_yield_spread"])
        metrics["mortgage_30y"] = self._latest_frame_value(fred_series["mortgage_30y"])
        metrics["financial_conditions_index"] = self._latest_frame_value(fred_series["financial_conditions_index"])
        if any(not frame.empty for frame in fred_series.values()):
            sources_used.append("fred")

        bls_snapshot = self._fetch_bls_macro_snapshot()
        for key, value in bls_snapshot.items():
            if key in metrics:
                metrics[key] = self._as_float(value)
        if bls_snapshot:
            sources_used.append("bls")

        bea_snapshot = self._fetch_bea_macro_snapshot()
        for key, value in bea_snapshot.items():
            if key in metrics:
                metrics[key] = self._as_float(value)
        if bea_snapshot:
            sources_used.append("bea")

        treasury_snapshot = self._fetch_treasury_macro_snapshot()
        for key, value in treasury_snapshot.items():
            if key in metrics:
                metrics[key] = self._as_float(value)
        if treasury_snapshot:
            sources_used.append("treasury")

        eia_snapshot = self._fetch_eia_macro_snapshot()
        for key, value in eia_snapshot.items():
            if key in metrics:
                metrics[key] = self._as_float(value)
        if eia_snapshot:
            sources_used.append("eia")

        revisions = self._fetch_alfred_revision_snapshot()
        for key, value in revisions.items():
            if key in metrics:
                metrics[key] = self._as_float(value)
        if revisions:
            sources_used.append("alfred")

        positioning = self._fetch_cftc_cot_snapshot()
        for key, value in positioning.items():
            if key in metrics:
                metrics[key] = self._as_float(value)
        if positioning:
            sources_used.append("cftc")

        qualitative = self._fetch_fed_qualitative_snapshot()
        if qualitative:
            sources_used.append("fed_qualitative")

        short_flow = self._fetch_finra_short_volume_snapshot()
        for key, value in short_flow.items():
            if key in metrics:
                metrics[key] = self._as_float(value)
        if short_flow:
            sources_used.append("finra")

        fedwatch = self._fetch_cme_fedwatch_snapshot()
        for key, value in fedwatch.items():
            if key in metrics:
                metrics[key] = self._as_float(value)
        if fedwatch:
            sources_used.append("cme_fedwatch")

        structured_macro = self._fetch_nasdaq_data_link_snapshot()
        if structured_macro:
            sources_used.append("nasdaq_data_link")

        ex_us_macro = self._fetch_ex_us_macro_snapshot()
        if ex_us_macro:
            sources_used.extend(
                [
                    source_name
                    for source_name in ("ecb", "imf", "world_bank")
                    if source_name in (ex_us_macro.get("sources_used") or []) and source_name not in sources_used
                ]
            )

        global_event = self._fetch_gdelt_global_event_snapshot()
        if global_event:
            sources_used.extend(
                [
                    source_name
                    for source_name in ("gdelt_context", "gdelt_geo")
                    if source_name not in sources_used
                ]
            )

        if metrics["yield_spread_10y_2y"] is not None and metrics["yield_spread_10y_2y"] < 0:
            signals.append("yield_curve_inverted")
            highlights.append(f"2s10s inverted {metrics['yield_spread_10y_2y']:.2f}")
        if metrics["high_yield_spread"] is not None and metrics["high_yield_spread"] >= 4.5:
            signals.append("credit_spreads_wide")
            highlights.append(f"HY spread wide {metrics['high_yield_spread']:.2f}")
        if metrics["unemployment_rate"] is not None and metrics["unemployment_rate"] >= 4.2:
            signals.append("labor_softening")
            highlights.append(f"unemployment {metrics['unemployment_rate']:.2f}%")
        if metrics["payrolls_mom_k"] is not None and metrics["payrolls_mom_k"] < 125:
            signals.append("payroll_momentum_soft")
            highlights.append(f"payroll add {metrics['payrolls_mom_k']:.0f}k")
        if metrics["core_cpi_yoy"] is not None and metrics["core_cpi_yoy"] >= 3.3:
            signals.append("core_inflation_sticky")
            highlights.append(f"core CPI {metrics['core_cpi_yoy']:.2f}%")
        if metrics["core_pce_yoy"] is not None and metrics["core_pce_yoy"] >= 2.9:
            signals.append("core_pce_sticky")
            highlights.append(f"core PCE {metrics['core_pce_yoy']:.2f}%")
        if metrics["gdp_qoq_annualized"] is not None and metrics["gdp_qoq_annualized"] < 1.5:
            signals.append("bea_growth_softening")
            highlights.append(f"GDP {metrics['gdp_qoq_annualized']:.2f}%")
        if metrics["personal_spending_mom"] is not None and metrics["personal_spending_mom"] < 0:
            signals.append("consumer_spending_soft")
            highlights.append(f"spending {metrics['personal_spending_mom']:+.2f}%")
        if metrics["gasoline_usd_gal"] is not None and metrics["gasoline_usd_gal"] >= 3.7:
            signals.append("energy_price_pressure")
            highlights.append(f"gasoline {metrics['gasoline_usd_gal']:.2f}/gal")
        if metrics["operating_cash_balance_b"] is not None and metrics["operating_cash_balance_b"] <= 500:
            signals.append("treasury_cash_thin")
            highlights.append(f"TGA {metrics['operating_cash_balance_b']:.0f}B")
        if metrics["alfred_payroll_revision_k"] is not None and abs(metrics["alfred_payroll_revision_k"]) >= 40:
            signals.append("macro_revision_risk")
            highlights.append(f"ALFRED payroll rev {metrics['alfred_payroll_revision_k']:+.0f}k")
        if metrics["cftc_equity_net_pct_oi"] is not None and metrics["cftc_equity_net_pct_oi"] >= 15:
            signals.append("equity_positioning_crowded_long")
            highlights.append(f"CFTC equity {metrics['cftc_equity_net_pct_oi']:+.1f}% OI")
        if metrics["cftc_ust10y_net_pct_oi"] is not None and metrics["cftc_ust10y_net_pct_oi"] <= -10:
            signals.append("rates_positioning_defensive")
            highlights.append(f"CFTC UST10Y {metrics['cftc_ust10y_net_pct_oi']:+.1f}% OI")
        if metrics["finra_spy_short_volume_ratio"] is not None and metrics["finra_spy_short_volume_ratio"] >= 0.55:
            signals.append("short_flow_heavy")
            highlights.append(f"FINRA SPY short ratio {metrics['finra_spy_short_volume_ratio']:.2f}")
        if metrics["fedwatch_next_meeting_cut_prob_pct"] is not None and metrics["fedwatch_next_meeting_cut_prob_pct"] >= 60:
            signals.append("fedwatch_cut_bias_next_meeting")
            highlights.append(f"FedWatch next meeting cut {metrics['fedwatch_next_meeting_cut_prob_pct']:.0f}%")
        if metrics["fedwatch_next_meeting_hike_prob_pct"] is not None and metrics["fedwatch_next_meeting_hike_prob_pct"] >= 30:
            signals.append("fedwatch_hike_risk_next_meeting")
            highlights.append(f"FedWatch next meeting hike {metrics['fedwatch_next_meeting_hike_prob_pct']:.0f}%")
        if metrics["fedwatch_easing_12m_prob_pct"] is not None and metrics["fedwatch_easing_12m_prob_pct"] >= 70:
            signals.append("market_pricing_easing_cycle")
            highlights.append(f"FedWatch 12m easing {metrics['fedwatch_easing_12m_prob_pct']:.0f}%")
        if metrics["fedwatch_hike_12m_prob_pct"] is not None and metrics["fedwatch_hike_12m_prob_pct"] >= 35:
            signals.append("market_pricing_higher_for_longer")
            highlights.append(f"FedWatch 12m hike risk {metrics['fedwatch_hike_12m_prob_pct']:.0f}%")
        if global_event:
            risk_score = self._as_float(global_event.get("risk_score"))
            if risk_score is not None and risk_score >= 0.65:
                signals.append("geopolitical_risk_elevated")
            top_theme = str(global_event.get("top_theme") or "").strip()
            if top_theme:
                highlights.append(f"GDELT theme {top_theme}")
            top_location = str(global_event.get("top_location") or "").strip()
            if top_location:
                highlights.append(f"GEO hotspot {top_location}")
        if ex_us_macro:
            if self._as_float(ex_us_macro.get("ecb_inflation_yoy")) is not None and self._as_float(ex_us_macro.get("ecb_inflation_yoy")) >= 2.7:
                signals.append("eurozone_inflation_sticky")
            if self._as_float(ex_us_macro.get("world_bank_gdp_growth")) is not None and self._as_float(ex_us_macro.get("world_bank_gdp_growth")) < 2.5:
                signals.append("global_growth_softening")
            top_highlight = next((str(item) for item in (ex_us_macro.get("highlights") or []) if str(item).strip()), None)
            if top_highlight:
                highlights.append(top_highlight)
        fed_tone = str(qualitative.get("fed_tone") or "").strip()
        if fed_tone == "hawkish":
            signals.append("fed_tone_hawkish")
            title = next((str(item) for item in (qualitative.get("speech_titles") or []) if str(item).strip()), None)
            if title:
                highlights.append(f"Fed tone hawkish: {title}")
        elif fed_tone == "dovish":
            signals.append("fed_tone_dovish")
            title = next((str(item) for item in (qualitative.get("speech_titles") or []) if str(item).strip()), None)
            if title:
                highlights.append(f"Fed tone dovish: {title}")

        headline = "macro backdrop mixed"
        if {"yield_curve_inverted", "credit_spreads_wide"} & set(signals):
            headline = "macro backdrop defensive"
        elif {"labor_softening", "payroll_momentum_soft"} & set(signals):
            headline = "growth slowing under the surface"
        elif {"core_inflation_sticky", "energy_price_pressure"} & set(signals):
            headline = "inflation pressure still sticky"
        elif {"equity_positioning_crowded_long", "short_flow_heavy"} & set(signals):
            headline = "risk assets supported but positioning crowded"
        elif {"market_pricing_easing_cycle", "fedwatch_cut_bias_next_meeting"} & set(signals):
            headline = "rates market leaning dovish"
        elif {"market_pricing_higher_for_longer", "fedwatch_hike_risk_next_meeting"} & set(signals):
            headline = "rates market pricing hawkish risk"
        elif metrics["vix"] is not None and metrics["vix"] <= 18 and not signals:
            headline = "risk appetite constructive"

        return {
            "headline": headline,
            "signals": list(dict.fromkeys(signals)),
            "highlights": list(dict.fromkeys(highlights))[:6],
            "metrics": metrics,
            "revisions": revisions,
            "positioning": positioning,
            "qualitative": qualitative,
            "short_flow": short_flow,
            "fedwatch": fedwatch,
            "structured_macro": structured_macro,
            "global_event": global_event,
            "ex_us_macro": ex_us_macro,
            "sources_used": list(dict.fromkeys(sources_used)),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _get_macro_event_calendar_sync(self, days_ahead: int) -> list[MacroEvent]:
        events = self._fetch_macro_release_events_from_fred(days_ahead=days_ahead)
        events.extend(self._fetch_fomc_events(days_ahead=days_ahead))
        events.extend(self._fetch_global_macro_calendar_events(days_ahead=days_ahead))
        events.sort(key=lambda item: item.scheduled_at)
        return events[:12]

    def _get_macro_surprise_signals_sync(self) -> list[MacroSurpriseSignal]:
        upcoming_map = {
            event.event_key: event.scheduled_at
            for event in self._get_macro_event_calendar_sync(days_ahead=45)
        }
        consensus_rows = self._fetch_consensus_calendar_rows(days_back=45, days_ahead=14)
        signals: list[MacroSurpriseSignal] = []

        cpi_signal = self._build_yoy_macro_surprise_signal(
            event_key="cpi",
            event_name="CPI",
            category="inflation",
            source="fred",
            frame=self._fetch_fred_series_frame("CPIAUCSL"),
            next_event_at=upcoming_map.get("cpi"),
            threshold=0.15,
            detail_url="https://fred.stlouisfed.org/series/CPIAUCSL",
        )
        if cpi_signal is not None:
            signals.append(self._merge_consensus_into_signal(cpi_signal, consensus_rows.get("cpi")))

        bls_history = self._fetch_bls_series_history(("WPUFD4", "CES0000000001"), years=3)
        ppi_frame = self._rows_to_time_series_frame(bls_history.get("WPUFD4", []))
        ppi_signal = self._build_yoy_macro_surprise_signal(
            event_key="ppi",
            event_name="PPI",
            category="inflation",
            source="bls",
            frame=ppi_frame,
            next_event_at=upcoming_map.get("ppi"),
            threshold=0.2,
            detail_url="https://download.bls.gov/pub/time.series/wp/",
        )
        if ppi_signal is not None:
            signals.append(self._merge_consensus_into_signal(ppi_signal, consensus_rows.get("ppi")))

        payroll_signal = self._build_payroll_surprise_signal(
            rows=bls_history.get("CES0000000001", []),
            next_event_at=upcoming_map.get("nfp"),
        )
        if payroll_signal is not None:
            signals.append(self._merge_consensus_into_signal(payroll_signal, consensus_rows.get("nfp")))

        fomc_signal = self._build_fomc_surprise_signal(next_event_at=upcoming_map.get("fomc"))
        if fomc_signal is not None:
            signals.append(self._merge_consensus_into_signal(fomc_signal, consensus_rows.get("fomc")))

        order = {"cpi": 0, "ppi": 1, "nfp": 2, "fomc": 3}
        signals.sort(key=lambda item: order.get(item.event_key, 99))
        return signals

    def _get_macro_market_reactions_sync(self) -> list[MacroMarketReaction]:
        surprise_signals = self._get_macro_surprise_signals_sync()
        now = datetime.now(timezone.utc)
        interesting = [
            signal
            for signal in surprise_signals
            if signal.released_at is not None
            and (now - signal.released_at) <= timedelta(days=7)
            and signal.market_bias not in {None, "balanced"}
            and signal.surprise_label not in {"in_line", "steady", "insufficient_data"}
        ]
        reaction_specs = (
            ("SPY", "SPY"),
            ("QQQ", "QQQ"),
            ("TLT", "TLT"),
            ("DX-Y.NYB", "DXY"),
            ("^VIX", "VIX"),
        )
        reactions: list[MacroMarketReaction] = []
        for signal in interesting[:4]:
            event_time = signal.released_at
            if event_time is None:
                continue
            asset_moves: list[MacroReactionAssetMove] = []
            confirmations_5m: list[bool] = []
            confirmations_1h: list[bool] = []
            for ticker, label in reaction_specs:
                bars = self._get_history_sync(ticker, period="10d", interval="5m", limit=None)
                move_5m, move_1h = self._compute_post_event_moves(
                    bars=bars,
                    event_time=event_time,
                    minute_offsets=(5, 60),
                )
                expected_direction = self._expected_market_direction(signal.market_bias, label)
                confirmed_5m = self._move_matches_direction(move_5m, expected_direction)
                confirmed_1h = self._move_matches_direction(move_1h, expected_direction)
                if confirmed_5m is not None:
                    confirmations_5m.append(confirmed_5m)
                if confirmed_1h is not None:
                    confirmations_1h.append(confirmed_1h)
                asset_moves.append(
                    MacroReactionAssetMove(
                        ticker=ticker,
                        label=label,
                        move_5m_pct=move_5m,
                        move_1h_pct=move_1h,
                        expected_direction=expected_direction,
                        confirmed_5m=confirmed_5m,
                        confirmed_1h=confirmed_1h,
                    )
                )
            score_5m = round(sum(1 for item in confirmations_5m if item) / len(confirmations_5m), 2) if confirmations_5m else None
            score_1h = round(sum(1 for item in confirmations_1h if item) / len(confirmations_1h), 2) if confirmations_1h else None
            confirmation_label = self._classify_market_confirmation(score_5m, score_1h)
            rationale = [
                f"5m_confirm={score_5m}" if score_5m is not None else "5m_confirm=n/a",
                f"1h_confirm={score_1h}" if score_1h is not None else "1h_confirm=n/a",
                f"bias={signal.market_bias or 'n/a'}",
            ]
            if confirmation_label == "not_confirmed":
                rationale.append("macro surprise strong but cross-asset reaction did not confirm")
            reactions.append(
                MacroMarketReaction(
                    event_key=signal.event_key,
                    event_name=signal.event_name,
                    released_at=signal.released_at,
                    market_bias=signal.market_bias,
                    confirmation_label=confirmation_label,
                    confirmation_score_5m=score_5m,
                    confirmation_score_1h=score_1h,
                    rationale=tuple(rationale),
                    reactions=tuple(asset_moves),
                )
            )
        return reactions

    def _fetch_bea_macro_snapshot(self) -> dict[str, float]:
        if not self.bea_api_key:
            return {}
        snapshot: dict[str, float] = {}
        pce_rows = self._fetch_bea_table_rows(table_name="T20804", frequency="M")
        spending_rows = self._fetch_bea_table_rows(table_name="T20805", frequency="M")
        gdp_rows = self._fetch_bea_table_rows(table_name="T10105", frequency="Q")
        snapshot["pce_yoy"] = self._extract_bea_yoy_value(pce_rows, ("Personal consumption expenditures (PCE)", "Personal consumption expenditures"))
        snapshot["core_pce_yoy"] = self._extract_bea_yoy_value(pce_rows, ("Personal consumption expenditures excluding food and energy (chain-type price index)", "PCE excluding food and energy"))
        snapshot["personal_income_mom"] = self._extract_bea_pct_value(spending_rows, ("Personal income",))
        snapshot["personal_spending_mom"] = self._extract_bea_pct_value(spending_rows, ("Personal consumption expenditures",))
        snapshot["gdp_qoq_annualized"] = self._extract_bea_pct_value(gdp_rows, ("Gross domestic product",))
        return {key: value for key, value in snapshot.items() if value is not None}

    def _fetch_alfred_revision_snapshot(self) -> dict[str, float]:
        if not self.fred_api_key:
            return {}
        snapshot: dict[str, float] = {}
        payroll_revisions = self._fetch_alfred_vintage_delta("PAYEMS")
        cpi_revisions = self._fetch_alfred_vintage_delta("CPIAUCSL")
        if payroll_revisions is not None:
            snapshot["alfred_payroll_revision_k"] = payroll_revisions
        if cpi_revisions is not None:
            snapshot["alfred_cpi_revision_pct"] = cpi_revisions
        return snapshot

    def _fetch_cftc_cot_snapshot(self) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        financial_text = self._fetch_text_url("https://www.cftc.gov/dea/options/financial_lof.htm")
        other_text = self._fetch_text_url("https://www.cftc.gov/dea/options/other_lof.htm")
        if financial_text:
            equity_value = self._extract_cftc_financial_net_pct_oi(
                financial_text,
                market_label="E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
            )
            ust10y_value = self._extract_cftc_financial_net_pct_oi(
                financial_text,
                market_label="10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
            )
            usd_value = self._extract_cftc_financial_net_pct_oi(
                financial_text,
                market_label="USD INDEX - ICE FUTURES U.S.",
            )
            if equity_value is not None:
                snapshot["cftc_equity_net_pct_oi"] = equity_value
            if ust10y_value is not None:
                snapshot["cftc_ust10y_net_pct_oi"] = ust10y_value
            if usd_value is not None:
                snapshot["cftc_usd_net_pct_oi"] = usd_value
        if other_text:
            gold_value = self._extract_cftc_other_net_pct_oi(
                other_text,
                market_label="GOLD - COMMODITY EXCHANGE INCORPORATED",
            )
            if gold_value is not None:
                snapshot["cftc_gold_net_pct_oi"] = gold_value
        return snapshot

    def _fetch_fed_qualitative_snapshot(self) -> dict[str, Any]:
        speeches_page = self._fetch_text_url("https://www.federalreserve.gov/newsevents/speeches-testimony.htm")
        beige_page = self._fetch_text_url("https://www.federalreserve.gov/monetarypolicy/publications/beige-book-default.htm")
        h41_page = self._fetch_text_url("https://www.federalreserve.gov/releases/H41/default.htm")
        speech_titles = self._extract_fed_speech_titles(speeches_page)
        fed_tone = self._score_fed_titles_tone(speech_titles)
        beige_release = self._extract_first_regex(beige_page, r"([A-Z][a-z]+ \d{1,2}, \d{4})")
        h41_last_update = self._extract_first_regex(h41_page, r"Last Update:\s*([A-Z][a-z]+ \d{2}, \d{4})")
        snapshot: dict[str, Any] = {
            "fed_tone": fed_tone,
            "speech_titles": speech_titles[:4],
            "beige_book_release": beige_release,
            "h41_last_update": h41_last_update,
        }
        return {
            key: value
            for key, value in snapshot.items()
            if value is not None and value != "" and value != []
        }

    def _fetch_finra_short_volume_snapshot(self) -> dict[str, float]:
        latest_rows = self._fetch_latest_finra_short_volume_rows(("SPY", "QQQ"))
        snapshot: dict[str, float] = {}
        for ticker in ("SPY", "QQQ"):
            row = latest_rows.get(ticker)
            if not row:
                continue
            short_volume = self._as_float(row.get("ShortVolume")) or 0.0
            short_exempt = self._as_float(row.get("ShortExemptVolume")) or 0.0
            total_volume = self._as_float(row.get("TotalVolume"))
            if total_volume in {None, 0}:
                continue
            ratio = round((short_volume + short_exempt) / total_volume, 4)
            snapshot[f"finra_{ticker.casefold()}_short_volume_ratio"] = ratio
        return snapshot

    def _fetch_cme_fedwatch_snapshot(self) -> dict[str, Any]:
        if not self.cme_fedwatch_api_url:
            return {}
        headers = {"User-Agent": "InvestAdvisorBot/0.2"}
        if self.cme_fedwatch_api_key:
            headers["Authorization"] = f"Bearer {self.cme_fedwatch_api_key}"
            headers["apikey"] = self.cme_fedwatch_api_key
            headers["x-api-key"] = self.cme_fedwatch_api_key
        try:
            response = self._get_http_client().get(
                self.cme_fedwatch_api_url,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("CME FedWatch request failed: {}", exc)
            return {}
        if not isinstance(payload, Mapping):
            return {}
        meeting_rows = self._extract_fedwatch_meeting_rows(payload)
        next_meeting = next((item for item in meeting_rows if isinstance(item, Mapping)), None)
        snapshot: dict[str, Any] = {}
        if next_meeting is not None:
            snapshot["next_meeting"] = str(next_meeting.get("meeting") or next_meeting.get("date") or "").strip() or None
            snapshot["fedwatch_next_meeting_cut_prob_pct"] = self._extract_probability_from_mapping(next_meeting, ("cut", "lower"))
            snapshot["fedwatch_next_meeting_hold_prob_pct"] = self._extract_probability_from_mapping(next_meeting, ("hold", "unchanged"))
            snapshot["fedwatch_next_meeting_hike_prob_pct"] = self._extract_probability_from_mapping(next_meeting, ("hike", "raise"))
            snapshot["fedwatch_target_rate_mid_pct"] = self._extract_target_mid_from_mapping(next_meeting)
        snapshot["fedwatch_easing_12m_prob_pct"] = self._extract_probability_from_mapping(
            payload,
            ("easing_12m", "easing", "cut_12m", "cuts", "lower_12m"),
        )
        snapshot["fedwatch_hike_12m_prob_pct"] = self._extract_probability_from_mapping(
            payload,
            ("hike_12m", "hiking", "higher_for_longer", "raise_12m"),
        )
        return {
            key: value
            for key, value in snapshot.items()
            if value is not None and value != ""
        }

    def _fetch_nasdaq_data_link_snapshot(self) -> dict[str, Any]:
        if (
            not self.nasdaq_data_link_api_key
            or not self.nasdaq_data_link_datasets
            or self._nasdaq_data_link_disabled
        ):
            return {}
        snapshot: dict[str, Any] = {}
        for dataset_code in self.nasdaq_data_link_datasets:
            try:
                response = self._get_http_client().get(
                    f"{self.nasdaq_data_link_base_url}/datasets/{dataset_code}.json",
                    params={"api_key": self.nasdaq_data_link_api_key, "rows": 1, "order": "desc"},
                    headers={"User-Agent": "InvestAdvisorBot/0.2"},
                    follow_redirects=True,
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code == 403:
                    self._nasdaq_data_link_disabled = True
                    self._nasdaq_data_link_warning = "nasdaq data link access forbidden for current account or environment"
                    logger.warning(
                        "Nasdaq Data Link disabled after HTTP 403 for {}",
                        dataset_code,
                    )
                    return {}
                logger.warning("Nasdaq Data Link request failed for {}: {}", dataset_code, exc)
                continue
            except Exception as exc:
                logger.warning("Nasdaq Data Link request failed for {}: {}", dataset_code, exc)
                continue
            dataset_payload = payload.get("dataset_data") if isinstance(payload, Mapping) and isinstance(payload.get("dataset_data"), Mapping) else payload.get("dataset") if isinstance(payload, Mapping) and isinstance(payload.get("dataset"), Mapping) else None
            if not isinstance(dataset_payload, Mapping):
                continue
            columns = dataset_payload.get("column_names")
            rows = dataset_payload.get("data")
            if not isinstance(columns, list) or not isinstance(rows, list) or not rows:
                continue
            latest_row = rows[0] if isinstance(rows[0], list) else None
            if not isinstance(latest_row, list):
                continue
            row_map = {str(column): latest_row[index] for index, column in enumerate(columns) if index < len(latest_row)}
            numeric_candidates = [
                (column_name, self._as_float(row_map.get(column_name)))
                for column_name in reversed([str(column) for column in columns[1:]])
            ]
            value_column, numeric_value = next(
                ((name, value) for name, value in numeric_candidates if value is not None),
                (None, None),
            )
            snapshot[dataset_code] = {
                "date": row_map.get(columns[0]) if columns else None,
                "value": numeric_value,
                "value_column": value_column,
                "row": row_map,
            }
        return snapshot

    def _fetch_ex_us_macro_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        highlights: list[str] = []
        sources_used: list[str] = []

        ecb_snapshot = self._fetch_ecb_ex_us_macro_snapshot()
        if ecb_snapshot:
            snapshot.update(ecb_snapshot)
            sources_used.append("ecb")
            inflation = self._as_float(ecb_snapshot.get("ecb_inflation_yoy"))
            if inflation is not None:
                highlights.append(f"ECB inflation {inflation:.2f}%")

        imf_snapshot = self._fetch_imf_ex_us_macro_snapshot()
        if imf_snapshot:
            snapshot.update(imf_snapshot)
            sources_used.append("imf")
            growth = self._as_float(imf_snapshot.get("imf_global_growth_pct"))
            if growth is not None:
                highlights.append(f"IMF global growth {growth:.2f}%")

        world_bank_snapshot = self._fetch_world_bank_ex_us_macro_snapshot()
        if world_bank_snapshot:
            snapshot.update(world_bank_snapshot)
            sources_used.append("world_bank")
            growth = self._as_float(world_bank_snapshot.get("world_bank_gdp_growth"))
            if growth is not None:
                highlights.append(f"WB ex-US growth {growth:.2f}%")

        if not snapshot:
            return {}
        snapshot["sources_used"] = sources_used
        snapshot["highlights"] = highlights[:4]
        return snapshot

    def _fetch_ecb_ex_us_macro_snapshot(self) -> dict[str, Any]:
        if not self.ecb_series_map or not self.ecb_api_base_url:
            return {}
        snapshot: dict[str, Any] = {}
        for label, series_key in self.ecb_series_map.items():
            try:
                response = self._get_http_client().get(
                    f"{self.ecb_api_base_url}/{series_key}",
                    params={"format": "jsondata", "lastNObservations": 1},
                    headers={"User-Agent": "InvestAdvisorBot/0.2"},
                    follow_redirects=True,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                logger.warning("ECB request failed for {}: {}", series_key, exc)
                continue
            value = self._extract_ecb_observation_value(payload)
            if value is not None:
                snapshot[f"ecb_{label}"] = value
        return snapshot

    def _fetch_imf_ex_us_macro_snapshot(self) -> dict[str, Any]:
        if not self.imf_series_map or not self.imf_api_base_url:
            return {}
        snapshot: dict[str, Any] = {}
        for label, series_key in self.imf_series_map.items():
            url = self.imf_api_base_url.format(series=series_key)
            try:
                response = self._get_http_client().get(
                    url,
                    headers={"User-Agent": "InvestAdvisorBot/0.2"},
                    follow_redirects=True,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                logger.warning("IMF request failed for {}: {}", series_key, exc)
                continue
            value = self._extract_nested_numeric_value(payload)
            if value is not None:
                snapshot[f"imf_{label}"] = value
        return snapshot

    def _fetch_world_bank_ex_us_macro_snapshot(self) -> dict[str, Any]:
        if not self.world_bank_indicator_map or not self.world_bank_countries:
            return {}
        snapshot: dict[str, Any] = {}
        countries = ";".join(self.world_bank_countries)
        for label, indicator_code in self.world_bank_indicator_map.items():
            try:
                response = self._get_http_client().get(
                    f"{self.world_bank_api_base_url}/country/{countries}/indicator/{indicator_code}",
                    params={"format": "json", "per_page": 8},
                    headers={"User-Agent": "InvestAdvisorBot/0.2"},
                    follow_redirects=True,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                logger.warning("World Bank request failed for {}: {}", indicator_code, exc)
                continue
            value = self._extract_world_bank_indicator_value(payload)
            if value is not None:
                snapshot[f"world_bank_{label}"] = value
        return snapshot

    @staticmethod
    def _extract_ecb_observation_value(payload: Any) -> float | None:
        if not isinstance(payload, Mapping):
            return None
        data_sets = payload.get("dataSets")
        if not isinstance(data_sets, list) or not data_sets:
            return None
        first = data_sets[0]
        if not isinstance(first, Mapping):
            return None
        series_map = first.get("series")
        if not isinstance(series_map, Mapping):
            return None
        for series_payload in series_map.values():
            if not isinstance(series_payload, Mapping):
                continue
            observations = series_payload.get("observations")
            if not isinstance(observations, Mapping):
                continue
            for observation in observations.values():
                if isinstance(observation, list) and observation:
                    try:
                        return float(observation[0])
                    except (TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def _extract_world_bank_indicator_value(payload: Any) -> float | None:
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return None
        for row in payload[1]:
            if not isinstance(row, Mapping):
                continue
            value = row.get("value")
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_nested_numeric_value(payload: Any) -> float | None:
        if isinstance(payload, Mapping):
            for value in payload.values():
                extracted = MarketDataClient._extract_nested_numeric_value(value)
                if extracted is not None:
                    return extracted
            return None
        if isinstance(payload, list):
            for value in payload:
                extracted = MarketDataClient._extract_nested_numeric_value(value)
                if extracted is not None:
                    return extracted
            return None
        try:
            return float(payload)
        except (TypeError, ValueError):
            return None

    def _fetch_gdelt_global_event_snapshot(self) -> dict[str, Any]:
        query = self.gdelt_query.strip()
        if not query:
            return {}
        cache_key = f"gdelt::{query.casefold()}::{self.gdelt_max_records}"
        with self._cache_lock:
            if cache_key in self._global_event_cache:
                return dict(self._global_event_cache[cache_key] or {})
        context_payload = self._fetch_gdelt_context_snapshot(query)
        geo_payload = self._fetch_gdelt_geo_snapshot(query)
        articles = context_payload.get("articles") if isinstance(context_payload.get("articles"), list) else []
        locations = geo_payload.get("locations") if isinstance(geo_payload.get("locations"), list) else []
        themes = context_payload.get("themes") if isinstance(context_payload.get("themes"), list) else []
        risk_terms = {
            "war",
            "conflict",
            "missile",
            "tariff",
            "sanction",
            "shipping",
            "attack",
            "oil",
            "election",
            "cyber",
        }
        theme_hits = [
            str(item or "").strip()
            for item in themes
            if str(item or "").strip() and any(term in str(item).casefold() for term in risk_terms)
        ]
        article_titles = [
            str(item.get("title") or "").strip()
            for item in articles
            if isinstance(item, Mapping) and str(item.get("title") or "").strip()
        ]
        top_location = None
        if locations:
            top_location = str(locations[0].get("name") or locations[0].get("location") or "").strip() or None
        risk_score = min(
            1.0,
            (min(len(theme_hits), 5) * 0.14)
            + (min(len(article_titles), 6) * 0.05)
            + (0.2 if top_location else 0.0),
        )
        snapshot = {
            "query": query,
            "article_count": len(articles),
            "location_count": len(locations),
            "top_theme": theme_hits[0] if theme_hits else (themes[0] if themes else None),
            "top_location": top_location,
            "risk_score": round(risk_score, 2),
            "headlines": article_titles[:4],
            "themes": themes[:6],
            "locations": locations[:5],
        }
        normalized = {
            key: value
            for key, value in snapshot.items()
            if value is not None and value != "" and value != []
        }
        with self._cache_lock:
            self._global_event_cache[cache_key] = dict(normalized)
        return normalized

    def _fetch_gdelt_context_snapshot(self, query: str) -> dict[str, Any]:
        try:
            response = self._get_http_client().get(
                self.gdelt_context_base_url,
                params={
                    "query": query,
                    "mode": "ArtList",
                    "format": "json",
                    "sort": "DateDesc",
                    "maxrecords": self.gdelt_max_records,
                },
                headers={"User-Agent": "InvestAdvisorBot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("GDELT context request failed: {}", exc)
            return {}
        articles_root = payload.get("articles") if isinstance(payload, Mapping) else None
        if isinstance(articles_root, Mapping):
            articles = articles_root.get("article")
        else:
            articles = articles_root
        if isinstance(articles, Mapping):
            articles = [articles]
        if not isinstance(articles, list):
            articles = []
        themes: list[str] = []
        normalized_articles: list[dict[str, Any]] = []
        for item in articles[: self.gdelt_max_records]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or item.get("seendate") or "").strip()
            if not title:
                continue
            source = str(item.get("domain") or item.get("sourcecountry") or "").strip() or None
            normalized_articles.append(
                {
                    "title": title,
                    "url": str(item.get("url") or "").strip() or None,
                    "domain": source,
                    "tone": self._as_float(item.get("tone")),
                    "seendate": str(item.get("seendate") or "").strip() or None,
                }
            )
            lower_title = title.casefold()
            for token in re.split(r"[^a-z0-9_]+", lower_title):
                token = token.strip()
                if len(token) >= 5 and token not in themes:
                    themes.append(token)
                if len(themes) >= 12:
                    break
        return {"articles": normalized_articles, "themes": themes}

    def _fetch_gdelt_geo_snapshot(self, query: str) -> dict[str, Any]:
        try:
            response = self._get_http_client().get(
                self.gdelt_geo_base_url,
                params={
                    "query": query,
                    "mode": "PointCloud",
                    "format": "json",
                    "maxrecords": self.gdelt_max_records,
                },
                headers={"User-Agent": "InvestAdvisorBot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("GDELT GEO request failed: {}", exc)
            return {}
        features = payload.get("features") if isinstance(payload, Mapping) and isinstance(payload.get("features"), list) else []
        locations: list[dict[str, Any]] = []
        for feature in features[: self.gdelt_max_records]:
            if not isinstance(feature, Mapping):
                continue
            props = feature.get("properties") if isinstance(feature.get("properties"), Mapping) else feature
            geometry = feature.get("geometry") if isinstance(feature.get("geometry"), Mapping) else {}
            coordinates = geometry.get("coordinates") if isinstance(geometry.get("coordinates"), list) else []
            locations.append(
                {
                    "name": str(props.get("name") or props.get("LocationName") or props.get("label") or "").strip() or None,
                    "lat": self._as_float(coordinates[1]) if len(coordinates) >= 2 else self._as_float(props.get("lat")),
                    "lon": self._as_float(coordinates[0]) if len(coordinates) >= 2 else self._as_float(props.get("lon")),
                    "count": self._as_int(props.get("count") or props.get("value")),
                }
            )
        return {"locations": [item for item in locations if item.get("name")]}

    def _fetch_latest_finra_short_volume_rows(self, tickers: Sequence[str]) -> dict[str, dict[str, Any]]:
        wanted = {ticker.upper() for ticker in tickers if ticker.strip()}
        if not wanted:
            return {}
        today = datetime.now(timezone.utc).date()
        for days_back in range(0, 10):
            trade_date = today - timedelta(days=days_back)
            if trade_date.weekday() >= 5:
                continue
            url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{trade_date:%Y%m%d}.txt"
            text = self._fetch_text_url(url)
            if not text:
                continue
            try:
                frame = pd.read_csv(StringIO(text), sep="|")
            except Exception:
                continue
            if frame.empty or "Symbol" not in frame.columns:
                continue
            rows: dict[str, dict[str, Any]] = {}
            for _, row in frame.iterrows():
                symbol = str(row.get("Symbol") or "").strip().upper()
                if symbol in wanted:
                    rows[symbol] = {str(key): row.get(key) for key in frame.columns}
            if rows:
                return rows
        return {}

    def _fetch_finra_short_volume_signal(self, ticker: str) -> str | None:
        rows = self._fetch_latest_finra_short_volume_rows((ticker.upper(),))
        row = rows.get(ticker.upper())
        if not row:
            return None
        short_volume = self._as_float(row.get("ShortVolume")) or 0.0
        short_exempt = self._as_float(row.get("ShortExemptVolume")) or 0.0
        total_volume = self._as_float(row.get("TotalVolume"))
        if total_volume in {None, 0}:
            return None
        ratio = (short_volume + short_exempt) / total_volume
        if ratio >= 0.6:
            return "elevated_short_pressure"
        if ratio <= 0.4:
            return "light_short_pressure"
        return "balanced_short_pressure"

    def _fetch_bea_table_rows(self, *, table_name: str, frequency: str) -> list[dict[str, Any]]:
        try:
            response = self._get_http_client().get(
                "https://apps.bea.gov/api/data",
                params={
                    "UserID": self.bea_api_key,
                    "method": "GetData",
                    "datasetname": "NIPA",
                    "TableName": table_name,
                    "Frequency": frequency,
                    "Year": "X",
                    "ResultFormat": "JSON",
                },
                headers={"User-Agent": "InvestAdvisorBot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("BEA request failed for {}: {}", table_name, exc)
            return []
        rows = (((payload or {}).get("BEAAPI") or {}).get("Results") or {}).get("Data")
        if not isinstance(rows, list):
            return []
        return [dict(item) for item in rows if isinstance(item, Mapping)]

    def _fetch_fred_vintage_dates(self, series_id: str, *, limit: int = 2) -> list[str]:
        if not self.fred_api_key:
            return []
        try:
            response = self._get_http_client().get(
                "https://api.stlouisfed.org/fred/series/vintagedates",
                params={"series_id": series_id, "api_key": self.fred_api_key, "file_type": "json"},
                headers={"User-Agent": "InvestAdvisorBot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("ALFRED vintagedates request failed for {}: {}", series_id, exc)
            return []
        dates = payload.get("vintage_dates") if isinstance(payload, Mapping) else None
        if not isinstance(dates, list):
            return []
        return [str(item).strip() for item in dates[-max(2, limit):] if str(item).strip()]

    def _fetch_fred_observation_for_vintage(self, series_id: str, vintage_date: str) -> float | None:
        try:
            response = self._get_http_client().get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "realtime_start": vintage_date,
                    "realtime_end": vintage_date,
                    "sort_order": "desc",
                    "limit": 1,
                },
                headers={"User-Agent": "InvestAdvisorBot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("ALFRED observations request failed for {} @ {}: {}", series_id, vintage_date, exc)
            return None
        observations = payload.get("observations") if isinstance(payload, Mapping) else None
        if not isinstance(observations, list) or not observations:
            return None
        return self._as_float(observations[0].get("value"))

    def _fetch_alfred_vintage_delta(self, series_id: str) -> float | None:
        vintage_dates = self._fetch_fred_vintage_dates(series_id, limit=2)
        if len(vintage_dates) < 2:
            return None
        previous = self._fetch_fred_observation_for_vintage(series_id, vintage_dates[-2])
        latest = self._fetch_fred_observation_for_vintage(series_id, vintage_dates[-1])
        if previous is None or latest is None:
            return None
        if series_id == "CPIAUCSL":
            if previous == 0:
                return None
            return round(((latest - previous) / abs(previous)) * 100.0, 3)
        return round(latest - previous, 2)

    def _extract_bea_pct_value(self, rows: Sequence[Mapping[str, Any]], line_descriptions: Sequence[str]) -> float | None:
        for label in line_descriptions:
            matches = [row for row in rows if label.casefold() in str(row.get("LineDescription") or "").casefold()]
            if not matches:
                continue
            ordered = sorted(matches, key=lambda item: str(item.get("TimePeriod") or ""))
            for row in reversed(ordered):
                value = self._parse_percent_like_value(row.get("DataValue"))
                if value is not None:
                    return value
        return None

    def _extract_bea_yoy_value(self, rows: Sequence[Mapping[str, Any]], line_descriptions: Sequence[str]) -> float | None:
        for label in line_descriptions:
            matches = [row for row in rows if label.casefold() in str(row.get("LineDescription") or "").casefold()]
            if len(matches) < 13:
                continue
            ordered = sorted(matches, key=lambda item: str(item.get("TimePeriod") or ""))
            current = self._parse_percent_like_value(ordered[-1].get("DataValue"))
            prior = self._parse_percent_like_value(ordered[-13].get("DataValue"))
            if current is None or prior in {None, 0}:
                continue
            return round(((current - prior) / abs(prior)) * 100.0, 2)
        return None

    def _extract_cftc_financial_net_pct_oi(self, text: str, *, market_label: str) -> float | None:
        section = self._extract_cftc_section(text, market_label)
        if not section:
            return None
        open_interest = self._extract_first_regex(section, r"Open Interest is\s+([\d,]+)")
        positions_line = self._extract_first_regex(section, r"Positions\s+([\d,\s]+)")
        if open_interest is None or positions_line is None:
            return None
        values = [int(value.replace(",", "")) for value in re.findall(r"\d[\d,]*", positions_line)]
        if len(values) < 6:
            return None
        open_interest_value = self._as_float(open_interest.replace(",", ""))
        if open_interest_value in {None, 0}:
            return None
        asset_manager_long = values[3]
        asset_manager_short = values[4]
        return round(((asset_manager_long - asset_manager_short) / open_interest_value) * 100.0, 2)

    def _extract_cftc_other_net_pct_oi(self, text: str, *, market_label: str) -> float | None:
        section = self._extract_cftc_section(text, market_label)
        if not section:
            return None
        open_interest = self._extract_first_regex(section, r"Open Interest is\s+([\d,]+)")
        positions_line = self._extract_first_regex(section, r"Positions\s+([\d,\s]+)")
        if open_interest is None or positions_line is None:
            return None
        values = [int(value.replace(",", "")) for value in re.findall(r"\d[\d,]*", positions_line)]
        if len(values) < 8:
            return None
        open_interest_value = self._as_float(open_interest.replace(",", ""))
        if open_interest_value in {None, 0}:
            return None
        managed_money_long = values[6]
        managed_money_short = values[7]
        return round(((managed_money_long - managed_money_short) / open_interest_value) * 100.0, 2)

    @staticmethod
    def _extract_cftc_section(text: str, market_label: str) -> str:
        normalized_label = re.escape(market_label)
        pattern = re.compile(rf"{normalized_label}.*?(?=(?:[A-Z0-9&/\-\.\s]{{20,}}Open Interest is)|\Z)", re.S)
        match = pattern.search(text or "")
        return match.group(0) if match else ""

    def _extract_fed_speech_titles(self, text: str | None) -> list[str]:
        if not text:
            return []
        titles: list[str] = []
        pattern = re.compile(r'<a[^>]+href="[^"]*(?:speech|testimony)[^"]*"[^>]*>(.*?)</a>', re.I | re.S)
        for match in pattern.finditer(text):
            title = self._normalize_document_text(match.group(1))
            if title and title not in titles:
                titles.append(title)
        return titles[:6]

    @staticmethod
    def _score_fed_titles_tone(titles: Sequence[str]) -> str:
        hawkish_tokens = ("inflation", "higher for longer", "restrictive", "upside risk", "tight")
        dovish_tokens = ("employment risk", "slowing", "easing", "disinflation", "downside")
        score = 0
        for title in titles:
            normalized = str(title).casefold()
            score += sum(1 for token in hawkish_tokens if token in normalized)
            score -= sum(1 for token in dovish_tokens if token in normalized)
        if score >= 2:
            return "hawkish"
        if score <= -2:
            return "dovish"
        return "neutral"

    def _fetch_text_url(self, url: str) -> str | None:
        try:
            response = self._get_http_client().get(
                url,
                headers={"User-Agent": "InvestAdvisorBot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except Exception:
            return None

    @staticmethod
    def _extract_first_regex(text: str | None, pattern: str) -> str | None:
        match = re.search(pattern, text or "", re.I | re.S)
        if not match:
            return None
        return str(match.group(1) or "").strip() or None

    @staticmethod
    def _extract_fedwatch_meeting_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        candidates = []
        for key in ("meetings", "results", "data", "probabilities"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(item for item in value if isinstance(item, Mapping))
        return candidates

    def _extract_probability_from_mapping(self, payload: Mapping[str, Any], keywords: Sequence[str]) -> float | None:
        normalized_keywords = tuple(str(item).casefold() for item in keywords if str(item).strip())
        for key, value in payload.items():
            normalized_key = str(key).casefold()
            if isinstance(value, Mapping):
                nested = self._extract_probability_from_mapping(value, keywords)
                if nested is not None:
                    return nested
                continue
            if not any(keyword in normalized_key for keyword in normalized_keywords):
                continue
            numeric = self._as_float(value)
            if numeric is None:
                continue
            if 0 <= numeric <= 1:
                return round(numeric * 100.0, 2)
            if 0 <= numeric <= 100:
                return round(numeric, 2)
        return None

    def _extract_target_mid_from_mapping(self, payload: Mapping[str, Any]) -> float | None:
        low = None
        high = None
        for key, value in payload.items():
            normalized = str(key).casefold()
            if isinstance(value, Mapping):
                nested = self._extract_target_mid_from_mapping(value)
                if nested is not None:
                    return nested
                continue
            if "target" in normalized and "low" in normalized:
                low = self._as_float(value)
            elif "target" in normalized and "high" in normalized:
                high = self._as_float(value)
            elif normalized in {"mid", "target_mid", "rate_mid"}:
                numeric = self._as_float(value)
                if numeric is not None:
                    return round(numeric, 3)
        if low is not None and high is not None:
            return round((low + high) / 2.0, 3)
        return None

    @staticmethod
    def _parse_percent_like_value(value: Any) -> float | None:
        text = str(value or "").replace(",", "").replace("%", "").strip()
        if not text or text in {"--", "(NA)"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _build_yoy_macro_surprise_signal(
        self,
        *,
        event_key: str,
        event_name: str,
        category: str,
        source: str,
        frame: pd.DataFrame,
        next_event_at: datetime | None,
        threshold: float,
        detail_url: str | None,
    ) -> MacroSurpriseSignal | None:
        if frame.empty or "value" not in frame.columns:
            return None
        ordered = frame.sort_values("date").copy()
        ordered["yoy"] = pd.to_numeric(ordered["value"], errors="coerce").pct_change(12) * 100.0
        yoy_values = ordered.dropna(subset=["yoy"]).reset_index(drop=True)
        if yoy_values.empty:
            return None
        actual = self._as_float(yoy_values.iloc[-1].get("yoy"))
        baseline_window = yoy_values["yoy"].tail(4).tolist()
        expected = self._median_previous_values(baseline_window)
        surprise = None if actual is None or expected is None else round(actual - expected, 2)
        released_at = self._normalize_timestamp(yoy_values.iloc[-1].get("date"))
        label, direction, bias = self._classify_macro_surprise(
            event_key=event_key,
            surprise=surprise,
            threshold=threshold,
        )
        rationale = [
            f"actual={actual:.2f}%" if actual is not None else "actual=n/a",
            f"baseline={expected:.2f}%" if expected is not None else "baseline=n/a",
        ]
        if surprise is not None:
            rationale.append(f"surprise={surprise:+.2f}ppt")
        if next_event_at is not None:
            rationale.append(f"next={next_event_at.isoformat()}")
        return MacroSurpriseSignal(
            event_key=event_key,
            event_name=event_name,
            category=category,
            source=source,
            released_at=released_at,
            next_event_at=next_event_at,
            actual_value=actual,
            expected_value=expected,
            surprise_value=surprise,
            surprise_direction=direction,
            surprise_label=label,
            market_bias=bias,
            rationale=tuple(rationale[:4]),
            detail_url=detail_url,
            baseline_expected_value=expected,
            baseline_surprise_value=surprise,
            baseline_surprise_label=label,
        )

    def _build_payroll_surprise_signal(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        next_event_at: datetime | None,
    ) -> MacroSurpriseSignal | None:
        frame = self._rows_to_time_series_frame(rows)
        if frame.empty or len(frame) < 5:
            return None
        ordered = frame.sort_values("date").copy()
        ordered["mom_k"] = pd.to_numeric(ordered["value"], errors="coerce").diff()
        mom_values = ordered.dropna(subset=["mom_k"]).reset_index(drop=True)
        if mom_values.empty:
            return None
        actual = self._as_float(mom_values.iloc[-1].get("mom_k"))
        baseline_window = mom_values["mom_k"].tail(4).tolist()
        expected = self._median_previous_values(baseline_window)
        surprise = None if actual is None or expected is None else round(actual - expected, 1)
        released_at = self._normalize_timestamp(mom_values.iloc[-1].get("date"))
        label, direction, bias = self._classify_macro_surprise(
            event_key="nfp",
            surprise=surprise,
            threshold=75.0,
        )
        rationale = [
            f"actual={actual:.0f}k" if actual is not None else "actual=n/a",
            f"baseline={expected:.0f}k" if expected is not None else "baseline=n/a",
        ]
        if surprise is not None:
            rationale.append(f"surprise={surprise:+.0f}k")
        revision_k = self._fetch_bls_macro_snapshot().get("payrolls_revision_k")
        if revision_k is not None:
            rationale.append(f"revision={revision_k:+.0f}k")
        return MacroSurpriseSignal(
            event_key="nfp",
            event_name="NFP",
            category="labor",
            source="bls",
            released_at=released_at,
            next_event_at=next_event_at,
            actual_value=actual,
            expected_value=expected,
            surprise_value=surprise,
            surprise_direction=direction,
            surprise_label=label,
            market_bias=bias,
            rationale=tuple(rationale[:4]),
            detail_url="https://download.bls.gov/pub/time.series/ce/",
            baseline_expected_value=expected,
            baseline_surprise_value=surprise,
            baseline_surprise_label=label,
        )

    def _build_fomc_surprise_signal(self, *, next_event_at: datetime | None) -> MacroSurpriseSignal | None:
        statement_urls = self._fetch_recent_fomc_statement_urls(limit=2)
        if len(statement_urls) < 2:
            return None
        latest_url, previous_url = statement_urls[0], statement_urls[1]
        latest_text = self._fetch_fomc_statement_text(latest_url)
        previous_text = self._fetch_fomc_statement_text(previous_url)
        if not latest_text or not previous_text:
            return None
        actual = float(self._score_fomc_statement(latest_text))
        expected = float(self._score_fomc_statement(previous_text))
        surprise = round(actual - expected, 2)
        label, direction, bias = self._classify_macro_surprise(
            event_key="fomc",
            surprise=surprise,
            threshold=1.0,
        )
        released_at = self._parse_datetime(self._extract_fomc_statement_date(latest_url))
        rationale = (
            f"latest_tone={actual:+.0f}",
            f"prior_tone={expected:+.0f}",
            f"shift={surprise:+.0f}",
            f"next={next_event_at.isoformat()}" if next_event_at is not None else "next=n/a",
        )
        return MacroSurpriseSignal(
            event_key="fomc",
            event_name="FOMC",
            category="policy",
            source="federal_reserve",
            released_at=released_at,
            next_event_at=next_event_at,
            actual_value=actual,
            expected_value=expected,
            surprise_value=surprise,
            surprise_direction=direction,
            surprise_label=label,
            market_bias=bias,
            rationale=rationale,
            detail_url=latest_url,
            baseline_expected_value=expected,
            baseline_surprise_value=surprise,
            baseline_surprise_label=label,
        )

    def _fetch_macro_release_events_from_fred(self, *, days_ahead: int) -> list[MacroEvent]:
        release_specs = (
            ("Consumer Price Index", "cpi", "CPI", "inflation", "high"),
            ("Producer Price Index", "ppi", "PPI", "inflation", "high"),
            ("Employment Situation", "nfp", "NFP", "labor", "high"),
        )
        events: list[MacroEvent] = []
        for release_name, event_key, event_name, category, importance in release_specs:
            for event_date in self._fetch_fred_release_dates_by_name(release_name, days_ahead=days_ahead):
                hour = 8
                minute = 30
                scheduled_at = self._build_market_event_datetime(event_date, hour=hour, minute=minute)
                if scheduled_at is None:
                    continue
                events.append(
                    MacroEvent(
                        event_key=event_key,
                        event_name=event_name,
                        category=category,
                        source="fred_release_calendar",
                        scheduled_at=scheduled_at,
                        importance=importance,
                        status="scheduled",
                        source_url="https://fred.stlouisfed.org/docs/api/fred/release_dates.html",
                    )
                )
        return events

    def _fetch_fomc_events(self, *, days_ahead: int) -> list[MacroEvent]:
        years = {datetime.now(timezone.utc).year, (datetime.now(timezone.utc) + timedelta(days=days_ahead)).year}
        try:
            response = self._get_http_client().get(
                "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            text = response.text
        except Exception as exc:
            logger.warning("Failed to fetch FOMC calendar: {}", exc)
            return []

        now = datetime.now(timezone.utc)
        horizon = now + timedelta(days=max(3, days_ahead))
        events: list[MacroEvent] = []
        for year in sorted(years):
            section = self._extract_fomc_year_section(text, year)
            if not section:
                continue
            pattern = re.compile(
                r'fomc-meeting__month[^>]*>\s*<strong>(?P<month>[^<]+)</strong>.*?'
                r'fomc-meeting__date[^>]*>(?P<dates>[^<]+)</div>',
                re.I | re.S,
            )
            for match in pattern.finditer(section):
                meeting_date = self._parse_fomc_meeting_date(
                    year=year,
                    month_name=match.group("month"),
                    date_text=match.group("dates"),
                )
                if meeting_date is None:
                    continue
                scheduled_at = self._build_market_event_datetime(meeting_date, hour=14, minute=0)
                if scheduled_at is None or scheduled_at < now or scheduled_at > horizon:
                    continue
                events.append(
                    MacroEvent(
                        event_key="fomc",
                        event_name="FOMC",
                        category="policy",
                        source="federal_reserve",
                        scheduled_at=scheduled_at,
                        importance="critical",
                        status="scheduled",
                        source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                    )
                )
        events.sort(key=lambda item: item.scheduled_at)
        return events

    @staticmethod
    def _extract_fomc_year_section(text: str, year: int) -> str:
        pattern = re.compile(
            rf'{year}\s+FOMC Meetings.*?(?=(?:\d{{4}}\s+FOMC Meetings)|(?:Future Year:)|</div>\s*</div>\s*</div>\s*$)',
            re.I | re.S,
        )
        match = pattern.search(text)
        return match.group(0) if match else ""

    @staticmethod
    def _parse_fomc_meeting_date(*, year: int, month_name: str, date_text: str) -> datetime | None:
        month_number = None
        try:
            month_number = datetime.strptime(month_name.strip(), "%B").month
        except ValueError:
            return None
        normalized = re.sub(r"[^0-9\-]", "", date_text or "")
        if not normalized:
            return None
        parts = [part for part in normalized.split("-") if part]
        if not parts:
            return None
        try:
            meeting_day = int(parts[-1])
            return datetime(year, month_number, meeting_day, tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _build_market_event_datetime(value: str | datetime, *, hour: int, minute: int) -> datetime | None:
        if isinstance(value, datetime):
            base = value
        else:
            try:
                base = datetime.fromisoformat(str(value).strip())
            except ValueError:
                return None
        try:
            eastern = ZoneInfo("America/New_York")
            localized = datetime(base.year, base.month, base.day, hour, minute, tzinfo=eastern)
            return localized.astimezone(timezone.utc)
        except Exception:
            return datetime(base.year, base.month, base.day, hour, minute, tzinfo=timezone.utc)

    def _fetch_fred_series_frame(self, series_id: str) -> pd.DataFrame:
        try:
            response = self._get_http_client().get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv",
                params={"id": series_id},
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
        except Exception as exc:
            logger.warning("Failed to fetch FRED series {}: {}", series_id, exc)
            return pd.DataFrame()
        date_column = "DATE" if "DATE" in frame.columns else "observation_date" if "observation_date" in frame.columns else None
        value_column = series_id if series_id in frame.columns else next(
            (column for column in frame.columns if column != date_column),
            None,
        )
        if date_column is None or value_column is None:
            return pd.DataFrame()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        cleaned = frame.dropna(subset=[date_column, value_column]).sort_values(date_column)
        return cleaned.rename(columns={date_column: "date", value_column: "value"})[["date", "value"]]

    def _fetch_fred_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.fred_api_key:
            return None
        request_params = {
            "api_key": self.fred_api_key,
            "file_type": "json",
        }
        if params:
            request_params.update(dict(params))
        try:
            response = self._get_http_client().get(
                f"https://api.stlouisfed.org/fred/{endpoint.lstrip('/')}",
                params=request_params,
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Failed to fetch FRED endpoint {}: {}", endpoint, exc)
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def _fetch_fred_release_dates_by_name(self, release_name: str, *, days_ahead: int) -> list[str]:
        release_id = self._resolve_fred_release_id(release_name)
        if release_id is None:
            return []
        payload = self._fetch_fred_json(
            "release/dates",
            params={
                "release_id": release_id,
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc",
                "limit": 24,
            },
        )
        rows = payload.get("release_dates") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            return []
        today = datetime.now(timezone.utc).date()
        horizon = today + timedelta(days=max(3, days_ahead))
        dates: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            date_text = str(row.get("date") or "").strip()
            if not date_text:
                continue
            try:
                event_date = datetime.fromisoformat(date_text).date()
            except ValueError:
                continue
            if event_date < today or event_date > horizon:
                continue
            dates.append(date_text)
        return dates

    def _resolve_fred_release_id(self, release_name: str) -> int | None:
        cache_key = release_name.strip().casefold()
        with self._cache_lock:
            if cache_key in self._fred_release_cache:
                return self._fred_release_cache[cache_key]
        payload = self._fetch_fred_json("releases", params={"limit": 1000, "order_by": "name", "sort_order": "asc"})
        releases = payload.get("releases") if isinstance(payload, Mapping) else None
        release_id: int | None = None
        if isinstance(releases, list):
            exact = None
            partial = None
            for row in releases:
                if not isinstance(row, Mapping):
                    continue
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                normalized = name.casefold()
                if normalized == cache_key:
                    exact = row
                    break
                if cache_key in normalized and partial is None:
                    partial = row
            matched = exact or partial
            if isinstance(matched, Mapping):
                release_id = self._as_int(matched.get("id"))
        with self._cache_lock:
            self._fred_release_cache[cache_key] = release_id
        return release_id

    def _fetch_bls_series_history(
        self,
        series_ids: Sequence[str],
        *,
        years: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        endpoint = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        payload: dict[str, Any] = {
            "seriesid": [item for item in series_ids if str(item).strip()],
            "startyear": str(datetime.now(timezone.utc).year - max(1, years)),
            "endyear": str(datetime.now(timezone.utc).year),
        }
        if not payload["seriesid"]:
            return {}
        if self.bls_api_key:
            payload["registrationkey"] = self.bls_api_key
        try:
            response = self._get_http_client().post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json", "User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("Failed to fetch BLS series history: {}", exc)
            return {}
        results = data.get("Results") if isinstance(data, Mapping) else None
        series_items = results.get("series") if isinstance(results, Mapping) else None
        if not isinstance(series_items, list):
            return {}
        parsed: dict[str, list[dict[str, Any]]] = {}
        for item in series_items:
            if not isinstance(item, Mapping):
                continue
            series_id = str(item.get("seriesID") or "").strip()
            rows = item.get("data")
            if not series_id or not isinstance(rows, list):
                continue
            series_rows: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                period = str(row.get("period") or "").strip().upper()
                year_text = str(row.get("year") or "").strip()
                if not period.startswith("M") or period == "M13" or not year_text.isdigit():
                    continue
                month_text = period[1:]
                if not month_text.isdigit():
                    continue
                month = int(month_text)
                if month < 1 or month > 12:
                    continue
                value = self._as_float(row.get("value"))
                if value is None:
                    continue
                try:
                    date_value = datetime(int(year_text), month, 1, tzinfo=timezone.utc)
                except ValueError:
                    continue
                series_rows.append({"date": date_value, "value": value})
            series_rows.sort(key=lambda current: current["date"])
            parsed[series_id] = series_rows
        return parsed

    @staticmethod
    def _rows_to_time_series_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            [
                {
                    "date": row.get("date"),
                    "value": row.get("value"),
                }
                for row in rows
            ]
        )
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _median_previous_values(values: Sequence[Any]) -> float | None:
        cleaned = [float(item) for item in values if item is not None and not pd.isna(item)]
        if len(cleaned) < 2:
            return None
        history = cleaned[:-1]
        if not history:
            return None
        return round(float(pd.Series(history).median()), 2)

    @staticmethod
    def _classify_macro_surprise(
        *,
        event_key: str,
        surprise: float | None,
        threshold: float,
    ) -> tuple[str, str | None, str | None]:
        if surprise is None:
            return "insufficient_data", None, None
        if event_key in {"cpi", "ppi"}:
            if surprise >= threshold:
                return "hotter_than_baseline", "hotter", "rates_up_risk_off"
            if surprise <= -threshold:
                return "cooler_than_baseline", "cooler", "duration_supportive"
            return "in_line", "in_line", "balanced"
        if event_key == "nfp":
            if surprise >= threshold:
                return "stronger_than_baseline", "stronger", "growth_positive_rates_up"
            if surprise <= -threshold:
                return "weaker_than_baseline", "weaker", "growth_negative_duration_support"
            return "in_line", "in_line", "balanced"
        if event_key == "fomc":
            if surprise >= threshold:
                return "hawkish_shift", "hawkish", "defensive_rates_up"
            if surprise <= -threshold:
                return "dovish_shift", "dovish", "risk_on_duration_support"
            return "steady", "steady", "balanced"
        return "in_line", "in_line", "balanced"

    def _fetch_recent_fomc_statement_urls(self, *, limit: int = 2) -> list[str]:
        try:
            response = self._get_http_client().get(
                "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            text = response.text
        except Exception as exc:
            logger.warning("Failed to fetch recent FOMC statement URLs: {}", exc)
            return []
        pattern = re.compile(r'href="(?P<path>/newsevents/pressreleases/monetary(?P<date>\d{8})a\.htm)"', re.I)
        seen: set[str] = set()
        urls: list[tuple[str, str]] = []
        for match in pattern.finditer(text):
            url = urljoin("https://www.federalreserve.gov", match.group("path"))
            if url in seen:
                continue
            seen.add(url)
            urls.append((match.group("date"), url))
        urls.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in urls[: max(2, limit)]]

    def _fetch_fomc_statement_text(self, url: str) -> str | None:
        try:
            response = self._get_http_client().get(
                url,
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            return self._normalize_document_text(response.text)
        except Exception as exc:
            logger.warning("Failed to fetch FOMC statement text for {}: {}", url, exc)
            return None

    @staticmethod
    def _score_fomc_statement(text: str) -> int:
        normalized = text.casefold()
        hawkish_terms = (
            "inflation remains somewhat elevated",
            "inflation remains elevated",
            "job gains have remained solid",
            "labor market conditions remain solid",
            "prepared to adjust the stance of monetary policy as appropriate if risks emerge",
            "maximum employment and inflation",
            "highly attentive to inflation risks",
        )
        dovish_terms = (
            "economic activity has softened",
            "job gains have slowed",
            "labor market conditions have eased",
            "inflation has eased",
            "uncertainty around the economic outlook has increased",
            "risks to achieving its employment and inflation goals are roughly in balance",
            "prepared to maintain the target range",
        )
        hawkish_score = sum(normalized.count(term) for term in hawkish_terms)
        dovish_score = sum(normalized.count(term) for term in dovish_terms)
        return hawkish_score - dovish_score

    @staticmethod
    def _extract_fomc_statement_date(url: str) -> str | None:
        match = re.search(r"monetary(\d{8})a\.htm", url or "", re.I)
        if not match:
            return None
        date_text = match.group(1)
        return f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:]}"

    def _fetch_consensus_calendar_rows(self, *, days_back: int, days_ahead: int) -> dict[str, Mapping[str, Any]]:
        rows = self._fetch_trading_economics_calendar(days_back=days_back, days_ahead=days_ahead)
        update_ids = self._fetch_trading_economics_update_ids(limit=100)
        if update_ids:
            rows.extend(self._fetch_trading_economics_calendar_ids(update_ids[:20]))
        if not rows:
            return {}
        matched: dict[str, list[Mapping[str, Any]]] = {"cpi": [], "ppi": [], "nfp": [], "fomc": []}
        for row in rows:
            event_key = self._classify_trading_economics_row(row)
            if event_key is None:
                continue
            matched[event_key].append(row)
        selected: dict[str, Mapping[str, Any]] = {}
        for event_key, event_rows in matched.items():
            released = [
                row for row in event_rows
                if self._parse_economic_value(row.get("Actual")) is not None
            ]
            candidate_pool = released or event_rows
            if not candidate_pool:
                continue
            candidate_pool.sort(
                key=lambda row: self._parse_datetime(row.get("Date")) or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            selected[event_key] = candidate_pool[0]
        return selected

    def _fetch_trading_economics_update_ids(self, *, limit: int = 50) -> list[str]:
        if self._trading_economics_calendar_disabled:
            return []
        credential = self.trading_economics_api_key or "guest:guest"
        try:
            response = self._get_http_client().get(
                "https://api.tradingeconomics.com/calendar/updates",
                params={"c": credential, "f": "json"},
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            if self._handle_trading_economics_calendar_error(exc):
                return []
            logger.warning("Failed to fetch Trading Economics updates: {}", exc)
            return []
        except Exception as exc:
            logger.warning("Failed to fetch Trading Economics updates: {}", exc)
            return []
        if not isinstance(payload, list):
            return []
        candidate_ids: list[str] = []
        for item in payload[: max(5, limit)]:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("Country") or "").strip().casefold() != "united states":
                continue
            event_key = self._classify_trading_economics_row(item)
            if event_key is None:
                continue
            calendar_id = str(item.get("CalendarId") or "").strip()
            if calendar_id and calendar_id not in candidate_ids:
                candidate_ids.append(calendar_id)
        return candidate_ids

    def _fetch_trading_economics_calendar_ids(self, calendar_ids: Sequence[str]) -> list[dict[str, Any]]:
        if self._trading_economics_calendar_disabled:
            return []
        filtered_ids = [str(item).strip() for item in calendar_ids if str(item).strip()]
        if not filtered_ids:
            return []
        credential = self.trading_economics_api_key or "guest:guest"
        try:
            response = self._get_http_client().get(
                f"https://api.tradingeconomics.com/calendar/calendarid/{','.join(filtered_ids)}",
                params={"c": credential, "f": "json"},
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            if self._handle_trading_economics_calendar_error(exc):
                return []
            logger.warning("Failed to fetch Trading Economics calendar IDs: {}", exc)
            return []
        except Exception as exc:
            logger.warning("Failed to fetch Trading Economics calendar IDs: {}", exc)
            return []
        return [dict(item) for item in payload] if isinstance(payload, list) else []

    def _fetch_trading_economics_calendar(self, *, days_back: int, days_ahead: int) -> list[dict[str, Any]]:
        if self._trading_economics_calendar_disabled:
            return []
        credential = self.trading_economics_api_key or "guest:guest"
        end_date = datetime.now(timezone.utc).date() + timedelta(days=max(1, days_ahead))
        start_date = datetime.now(timezone.utc).date() - timedelta(days=max(7, days_back))
        try:
            response = self._get_http_client().get(
                f"https://api.tradingeconomics.com/calendar/country/united states/{start_date.isoformat()}/{end_date.isoformat()}",
                params={"c": credential, "importance": 3, "f": "json"},
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            if self._handle_trading_economics_calendar_error(exc):
                return []
            logger.warning("Failed to fetch Trading Economics calendar: {}", exc)
            return []
        except Exception as exc:
            logger.warning("Failed to fetch Trading Economics calendar: {}", exc)
            return []
        return [dict(item) for item in payload] if isinstance(payload, list) else []

    def _fetch_trading_economics_calendar_for_countries(
        self,
        countries: Sequence[str],
        *,
        days_back: int,
        days_ahead: int,
        importance: int,
    ) -> list[dict[str, Any]]:
        if self._trading_economics_calendar_disabled:
            return []
        filtered_countries = [str(item).strip() for item in countries if str(item).strip()]
        if not filtered_countries:
            return []
        credential = self.trading_economics_api_key or "guest:guest"
        end_date = datetime.now(timezone.utc).date() + timedelta(days=max(1, days_ahead))
        start_date = datetime.now(timezone.utc).date() - timedelta(days=max(0, days_back))
        country_path = quote(",".join(filtered_countries), safe=",")
        try:
            response = self._get_http_client().get(
                f"https://api.tradingeconomics.com/calendar/country/{country_path}/{start_date.isoformat()}/{end_date.isoformat()}",
                params={"c": credential, "importance": max(1, int(importance)), "f": "json"},
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            if self._handle_trading_economics_calendar_error(exc):
                return []
            logger.warning("Failed to fetch Trading Economics global calendar: {}", exc)
            return []
        except Exception as exc:
            logger.warning("Failed to fetch Trading Economics global calendar: {}", exc)
            return []
        return [dict(item) for item in payload] if isinstance(payload, list) else []

    def _handle_trading_economics_calendar_error(self, exc: httpx.HTTPStatusError) -> bool:
        status_code = exc.response.status_code
        if status_code not in {403, 409}:
            return False
        if self._trading_economics_calendar_disabled:
            return True
        self._trading_economics_calendar_disabled = True
        body = str(exc.response.text or "").casefold()
        if "no access" in body or status_code == 403:
            self._trading_economics_warning = "trading economics calendar access restricted for current free plan"
        else:
            self._trading_economics_warning = "trading economics calendar disabled after repeated free-plan conflict responses"
        logger.warning("Trading Economics calendar endpoints disabled after HTTP {}: {}", status_code, self._trading_economics_warning)
        return True

    def _fetch_global_macro_calendar_events(self, *, days_ahead: int) -> list[MacroEvent]:
        rows = self._fetch_trading_economics_calendar_for_countries(
            self.global_macro_calendar_countries,
            days_back=0,
            days_ahead=days_ahead,
            importance=self.global_macro_calendar_importance,
        )
        if not rows:
            return []
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(days=max(3, days_ahead))
        events: list[MacroEvent] = []
        seen: set[tuple[str, str, datetime]] = set()
        for row in rows:
            scheduled_at = self._parse_datetime(row.get("Date"))
            if scheduled_at is None or scheduled_at < now - timedelta(hours=6) or scheduled_at > horizon:
                continue
            event_name = str(row.get("Event") or row.get("Category") or "").strip()
            country = str(row.get("Country") or "").strip() or None
            if not event_name or not country:
                continue
            key = (country.casefold(), event_name.casefold(), scheduled_at)
            if key in seen:
                continue
            seen.add(key)
            events.append(
                MacroEvent(
                    event_key=self._slugify_macro_event_key(country=country, event_name=event_name),
                    event_name=event_name,
                    category=self._normalize_macro_category(row.get("Category")),
                    source="trading_economics_global_calendar",
                    scheduled_at=scheduled_at,
                    importance=self._normalize_trading_economics_importance(row.get("Importance")),
                    status="scheduled" if self._parse_economic_value(row.get("Actual")) is None else "released",
                    source_url=self._resolve_trading_economics_detail_url(row),
                    country=country,
                    previous_value=self._parse_economic_value(row.get("Previous")),
                    forecast_value=self._parse_economic_value(row.get("Forecast") or row.get("TEForecast")),
                    actual_value=self._parse_economic_value(row.get("Actual")),
                )
            )
        events.sort(key=lambda item: item.scheduled_at)
        return events[:8]

    @staticmethod
    def _slugify_macro_event_key(*, country: str, event_name: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", f"{country}_{event_name}".casefold()).strip("_")
        return normalized[:80] or "macro_event"

    @staticmethod
    def _normalize_macro_category(value: Any) -> str:
        text = str(value or "").strip().casefold()
        if "inflation" in text or "price" in text:
            return "inflation"
        if "employment" in text or "payroll" in text or "labor" in text:
            return "labor"
        if "interest" in text or "rate" in text or "central bank" in text:
            return "policy"
        if "gdp" in text or "growth" in text:
            return "growth"
        return text or "macro"

    @staticmethod
    def _normalize_trading_economics_importance(value: Any) -> str:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            numeric = None
        if numeric is None:
            return "medium"
        if numeric >= 3:
            return "high"
        if numeric <= 1:
            return "low"
        return "medium"

    @staticmethod
    def _classify_trading_economics_row(row: Mapping[str, Any]) -> str | None:
        category = str(row.get("Category") or "").casefold()
        event = str(row.get("Event") or "").casefold()
        combined = f"{category} {event}"
        if "non farm payroll" in combined or "nonfarm payroll" in combined:
            return "nfp"
        if "producer prices" in combined or "producer price" in combined or "ppi" in combined:
            return "ppi"
        if "interest rate decision" in combined and ("fed" in combined or "federal reserve" in combined):
            return "fomc"
        if ("inflation rate" in combined or "consumer price" in combined or "cpi" in combined) and "producer" not in combined:
            return "cpi"
        return None

    def _merge_consensus_into_signal(
        self,
        signal: MacroSurpriseSignal,
        consensus_row: Mapping[str, Any] | None,
    ) -> MacroSurpriseSignal:
        if not isinstance(consensus_row, Mapping):
            return signal
        actual = self._parse_economic_value(consensus_row.get("Actual"))
        forecast = self._parse_economic_value(consensus_row.get("Forecast"))
        te_forecast = self._parse_economic_value(consensus_row.get("TEForecast"))
        consensus_expected = forecast if forecast is not None else te_forecast
        consensus_surprise = None if actual is None or consensus_expected is None else round(actual - consensus_expected, 2)
        threshold = 0.15 if signal.event_key == "cpi" else 0.2 if signal.event_key == "ppi" else 75.0 if signal.event_key == "nfp" else 0.12
        consensus_label, direction, bias = self._classify_macro_surprise(
            event_key=signal.event_key,
            surprise=consensus_surprise,
            threshold=threshold,
        )
        preferred_label = consensus_label if consensus_label not in {"insufficient_data"} else signal.surprise_label
        preferred_expected = consensus_expected if consensus_expected is not None else signal.expected_value
        preferred_surprise = consensus_surprise if consensus_surprise is not None else signal.surprise_value
        preferred_bias = bias if consensus_surprise is not None else signal.market_bias
        preferred_direction = direction if consensus_surprise is not None else signal.surprise_direction
        preferred_url = self._resolve_trading_economics_detail_url(consensus_row)
        rationale = list(signal.rationale)
        if consensus_expected is not None:
            rationale.append(f"consensus={consensus_expected}")
        if te_forecast is not None and te_forecast != consensus_expected:
            rationale.append(f"te={te_forecast}")
        if consensus_surprise is not None:
            rationale.append(f"cons_surprise={consensus_surprise:+.2f}")
        return replace(
            signal,
            source="trading_economics+baseline",
            actual_value=actual if actual is not None else signal.actual_value,
            expected_value=preferred_expected,
            surprise_value=preferred_surprise,
            surprise_direction=preferred_direction,
            surprise_label=preferred_label,
            market_bias=preferred_bias,
            rationale=tuple(rationale[:6]),
            detail_url=preferred_url or signal.detail_url,
            consensus_expected_value=consensus_expected,
            consensus_te_expected_value=te_forecast,
            consensus_surprise_value=consensus_surprise,
            consensus_surprise_label=consensus_label,
        )

    @staticmethod
    def _compute_post_event_moves(
        *,
        bars: Sequence[OhlcvBar],
        event_time: datetime,
        minute_offsets: tuple[int, int],
    ) -> tuple[float | None, float | None]:
        if not bars:
            return None, None
        sorted_bars = sorted(bars, key=lambda item: item.timestamp)
        base_bar = next((bar for bar in reversed(sorted_bars) if bar.timestamp <= event_time), None)
        if base_bar is None or base_bar.close == 0:
            return None, None
        target_5m = event_time + timedelta(minutes=minute_offsets[0])
        target_1h = event_time + timedelta(minutes=minute_offsets[1])
        bar_5m = next((bar for bar in sorted_bars if bar.timestamp >= target_5m), None)
        bar_1h = next((bar for bar in sorted_bars if bar.timestamp >= target_1h), None)
        move_5m = None if bar_5m is None else round(((bar_5m.close - base_bar.close) / base_bar.close) * 100.0, 2)
        move_1h = None if bar_1h is None else round(((bar_1h.close - base_bar.close) / base_bar.close) * 100.0, 2)
        return move_5m, move_1h

    @staticmethod
    def _expected_market_direction(market_bias: str | None, label: str) -> str:
        bias = (market_bias or "").strip().casefold()
        risk_off_map = {"SPY": "down", "QQQ": "down", "TLT": "down", "DXY": "up", "VIX": "up"}
        duration_support_map = {"SPY": "up", "QQQ": "up", "TLT": "up", "DXY": "down", "VIX": "down"}
        growth_negative_map = {"SPY": "down", "QQQ": "down", "TLT": "up", "DXY": "down", "VIX": "up"}
        growth_positive_map = {"SPY": "up", "QQQ": "up", "TLT": "down", "DXY": "up", "VIX": "down"}
        if "growth_negative" in bias:
            return growth_negative_map.get(label, "flat")
        if "growth_positive" in bias:
            return growth_positive_map.get(label, "flat")
        if "duration_support" in bias or "risk_on" in bias:
            return duration_support_map.get(label, "flat")
        if "risk_off" in bias or "defensive" in bias or "rates_up" in bias:
            return risk_off_map.get(label, "flat")
        return "flat"

    @staticmethod
    def _move_matches_direction(move_pct: float | None, direction: str) -> bool | None:
        if move_pct is None:
            return None
        if direction == "up":
            return move_pct > 0.05
        if direction == "down":
            return move_pct < -0.05
        return abs(move_pct) <= 0.1

    @staticmethod
    def _classify_market_confirmation(score_5m: float | None, score_1h: float | None) -> str:
        available = [value for value in (score_5m, score_1h) if value is not None]
        if not available:
            return "insufficient_data"
        average_score = sum(available) / len(available)
        if average_score >= 0.65:
            return "confirmed"
        if average_score <= 0.35:
            return "not_confirmed"
        return "mixed"

    @staticmethod
    def _resolve_trading_economics_detail_url(row: Mapping[str, Any]) -> str | None:
        url_path = str(row.get("URL") or "").strip()
        if not url_path:
            return None
        return urljoin("https://tradingeconomics.com", url_path)

    @staticmethod
    def _parse_economic_value(value: Any) -> float | None:
        text = str(value or "").strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return None
        normalized = text.replace(",", "").replace("%", "").replace("bps", "").replace("bp", "").strip()
        suffix = normalized[-1:].upper()
        if suffix in {"K", "M", "B"}:
            normalized = normalized[:-1]
        try:
            return round(float(normalized), 2)
        except ValueError:
            return None

    def _fetch_bls_macro_snapshot(self) -> dict[str, float | None]:
        endpoint = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        payload: dict[str, Any] = {
            "seriesid": [
                "CUUR0000SA0L1E",
                "WPUFD4",
                "CES0000000001",
                "LNS14000000",
            ],
            "startyear": str(datetime.now(timezone.utc).year - 2),
            "endyear": str(datetime.now(timezone.utc).year),
        }
        if self.bls_api_key:
            payload["registrationkey"] = self.bls_api_key
        try:
            response = self._get_http_client().post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json", "User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("Failed to fetch BLS macro snapshot: {}", exc)
            return {}
        results = data.get("Results") if isinstance(data, Mapping) else None
        series_items = results.get("series") if isinstance(results, Mapping) else None
        if not isinstance(series_items, list):
            return {}

        parsed: dict[str, list[dict[str, Any]]] = {}
        for item in series_items:
            if not isinstance(item, Mapping):
                continue
            series_id = str(item.get("seriesID") or "").strip()
            raw_rows = item.get("data")
            if not series_id or not isinstance(raw_rows, list):
                continue
            rows: list[dict[str, Any]] = []
            for row in raw_rows:
                if not isinstance(row, Mapping):
                    continue
                period = str(row.get("period") or "").strip()
                year = str(row.get("year") or "").strip()
                if not period.startswith("M") or not year.isdigit():
                    continue
                value = self._as_float(row.get("value"))
                month = self._as_int(period[1:])
                if value is None or month is None or month < 1 or month > 12:
                    continue
                rows.append(
                    {
                        "date": datetime(int(year), month, 1, tzinfo=timezone.utc),
                        "value": value,
                    }
                )
            rows.sort(key=lambda item: item["date"])
            parsed[series_id] = rows

        core_rows = parsed.get("CUUR0000SA0L1E", [])
        ppi_rows = parsed.get("WPUFD4", [])
        payroll_rows = parsed.get("CES0000000001", [])
        unemployment_rows = parsed.get("LNS14000000", [])
        return {
            "core_cpi_yoy": self._compute_yoy_from_rows(core_rows),
            "ppi_yoy": self._compute_yoy_from_rows(ppi_rows),
            "payrolls_mom_k": self._compute_diff_from_rows(payroll_rows),
            "unemployment_rate": self._latest_row_value(unemployment_rows),
        }

    def _fetch_treasury_macro_snapshot(self) -> dict[str, float | None]:
        avg_rates_rows = self._fetch_fiscal_data_rows(
            "v2/accounting/od/avg_interest_rates",
            params={"sort": "-record_date", "page[size]": 20},
        )
        operating_cash_rows = self._fetch_fiscal_data_rows(
            "v1/accounting/dts/operating_cash_balance",
            params={"sort": "-record_date", "page[size]": 20},
        )
        debt_rows = self._fetch_fiscal_data_rows(
            "v2/accounting/od/debt_to_penny",
            params={"sort": "-record_date", "page[size]": 2},
        )

        avg_interest_rate = None
        for row in avg_rates_rows:
            security_type = str(row.get("security_type_desc") or "").strip().casefold()
            security_desc = str(row.get("security_desc") or "").strip().casefold()
            if security_type == "marketable" and ("notes" in security_desc or "bonds" in security_desc):
                avg_interest_rate = self._as_float(row.get("avg_interest_rate_amt"))
                if avg_interest_rate is not None:
                    break

        operating_cash_balance = None
        for row in operating_cash_rows:
            account_type = str(row.get("account_type") or "").strip().casefold()
            if "opening balance" in account_type and "tga" in account_type:
                operating_cash_balance = self._as_float(row.get("open_today_bal"))
                if operating_cash_balance is not None:
                    operating_cash_balance /= 1_000.0
                    break

        public_debt_total = None
        if debt_rows:
            public_debt_total = self._as_float(debt_rows[0].get("tot_pub_debt_out_amt"))
            if public_debt_total is not None:
                public_debt_total /= 1_000_000_000_000.0

        return {
            "avg_interest_rate_pct": avg_interest_rate,
            "operating_cash_balance_b": operating_cash_balance,
            "public_debt_total_t": public_debt_total,
        }

    def _fetch_eia_macro_snapshot(self) -> dict[str, float | None]:
        series_map = {
            "wti_usd": "PET.RWTC.D",
            "brent_usd": "PET.RBRTE.D",
            "gasoline_usd_gal": "PET.EMM_EPMRR_PTE_NUS_DPG.W",
            "natgas_usd_mmbtu": "NG.RNGWHHD.D",
        }
        snapshot: dict[str, float | None] = {}
        for key, series_id in series_map.items():
            snapshot[key] = self._fetch_eia_latest_value(series_id)
        return snapshot

    def _fetch_fiscal_data_rows(self, dataset_path: str, *, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        endpoint = f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/{dataset_path.lstrip('/')}"
        headers = {"User-Agent": "invest-advisor-bot/0.2", "Accept": "application/json"}
        clients: list[httpx.Client] = [self._get_http_client()]
        try:
            clients.append(httpx.Client(timeout=self.http_timeout_seconds, verify=False))
        except Exception:
            pass
        for index, client in enumerate(clients):
            try:
                response = client.get(endpoint, params=dict(params), headers=headers, follow_redirects=True)
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") if isinstance(payload, Mapping) else None
                return list(data) if isinstance(data, list) else []
            except Exception as exc:
                if index == len(clients) - 1:
                    logger.warning("Failed to fetch Treasury dataset {}: {}", dataset_path, exc)
        return []

    def _fetch_eia_latest_value(self, series_id: str) -> float | None:
        api_key = self.eia_api_key or "DEMO_KEY"
        try:
            response = self._get_http_client().get(
                f"https://api.eia.gov/v2/seriesid/{series_id}",
                params={"api_key": api_key},
                headers={"User-Agent": "invest-advisor-bot/0.2"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Failed to fetch EIA series {}: {}", series_id, exc)
            return None
        data = ((payload or {}).get("response") or {}).get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list) or not data:
            return None
        return self._as_float(data[0].get("value"))

    @staticmethod
    def _latest_frame_value(frame: pd.DataFrame) -> float | None:
        if frame.empty:
            return None
        return MarketDataClient._as_float(frame.iloc[-1].get("value"))

    @staticmethod
    def _compute_yoy_from_frame(frame: pd.DataFrame) -> float | None:
        if frame.empty or len(frame) < 13:
            return None
        latest = MarketDataClient._as_float(frame.iloc[-1].get("value"))
        prior = MarketDataClient._as_float(frame.iloc[-13].get("value"))
        if latest is None or prior in {None, 0}:
            return None
        return round(((latest - prior) / prior) * 100.0, 2)

    @staticmethod
    def _compute_yoy_from_rows(rows: Sequence[Mapping[str, Any]]) -> float | None:
        if len(rows) < 13:
            return None
        latest = MarketDataClient._as_float(rows[-1].get("value"))
        prior = MarketDataClient._as_float(rows[-13].get("value"))
        if latest is None or prior in {None, 0}:
            return None
        return round(((latest - prior) / prior) * 100.0, 2)

    @staticmethod
    def _compute_diff_from_rows(rows: Sequence[Mapping[str, Any]]) -> float | None:
        if len(rows) < 2:
            return None
        latest = MarketDataClient._as_float(rows[-1].get("value"))
        prior = MarketDataClient._as_float(rows[-2].get("value"))
        if latest is None or prior is None:
            return None
        return round(latest - prior, 2)

    @staticmethod
    def _latest_row_value(rows: Sequence[Mapping[str, Any]]) -> float | None:
        if not rows:
            return None
        return MarketDataClient._as_float(rows[-1].get("value"))

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
        event = self._get_next_earnings_event_yfinance_sync(ticker)
        if event is not None:
            return event
        event = self._get_next_earnings_event_alpha_vantage_sync(ticker)
        if event is not None:
            return event
        return self._get_next_earnings_event_finnhub_sync(ticker)

    def _get_recent_earnings_result_sync(self, ticker: str, lookback_days: int) -> RecentEarningsResult | None:
        result = self._get_recent_earnings_result_yfinance_sync(ticker, lookback_days)
        if result is not None:
            return result
        return self._get_recent_earnings_result_alpha_vantage_sync(ticker, lookback_days)

    def _get_analyst_expectation_profile_sync(self, ticker: str) -> AnalystExpectationProfile | None:
        profile = self._get_analyst_expectation_profile_yfinance_sync(ticker)
        if profile is not None:
            return profile
        profile = self._get_analyst_expectation_profile_alpha_vantage_sync(ticker)
        if profile is not None:
            return profile
        return self._get_analyst_expectation_profile_finnhub_sync(ticker)

    def _get_next_earnings_event_yfinance_sync(self, ticker: str) -> EarningsEvent | None:
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

    def _get_recent_earnings_result_yfinance_sync(self, ticker: str, lookback_days: int) -> RecentEarningsResult | None:
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

    def _get_analyst_expectation_profile_yfinance_sync(self, ticker: str) -> AnalystExpectationProfile | None:
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

    def _get_next_earnings_event_alpha_vantage_sync(self, ticker: str) -> EarningsEvent | None:
        rows = self._fetch_alpha_vantage_csv(function="EARNINGS_CALENDAR", symbol=ticker.upper(), extra_params={"horizon": "3month"})
        if rows.empty:
            return None
        now = datetime.now(timezone.utc)
        for _, row in rows.iterrows():
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol and symbol != ticker.upper():
                continue
            earnings_at = self._parse_datetime(row.get("reportDate"))
            if earnings_at is None or earnings_at < now:
                continue
            return EarningsEvent(
                ticker=ticker.upper(),
                earnings_at=earnings_at,
                eps_estimate=self._as_float(row.get("estimate")),
                reported_eps=self._as_float(row.get("eps")),
                surprise_pct=self._as_float(row.get("surprisePercentage")),
            )
        return None

    def _get_recent_earnings_result_alpha_vantage_sync(self, ticker: str, lookback_days: int) -> RecentEarningsResult | None:
        payload = self._fetch_alpha_vantage_json(function="EARNINGS", symbol=ticker.upper())
        if not isinstance(payload, Mapping):
            return None
        quarterly = payload.get("quarterlyEarnings")
        if not isinstance(quarterly, list):
            return None
        now = datetime.now(timezone.utc)
        for item in quarterly:
            if not isinstance(item, Mapping):
                continue
            earnings_at = self._parse_datetime(item.get("reportedDate"))
            if earnings_at is None or earnings_at > now or (now - earnings_at).days > max(1, lookback_days):
                continue
            return RecentEarningsResult(
                ticker=ticker.upper(),
                earnings_at=earnings_at,
                eps_estimate=self._as_float(item.get("estimatedEPS")),
                reported_eps=self._as_float(item.get("reportedEPS")),
                surprise_pct=self._as_float(item.get("surprisePercentage")),
            )
        return None

    def _get_analyst_expectation_profile_alpha_vantage_sync(self, ticker: str) -> AnalystExpectationProfile | None:
        payload = self._fetch_alpha_vantage_json(function="EARNINGS_ESTIMATES", symbol=ticker.upper())
        if not isinstance(payload, Mapping):
            return None
        estimates = payload.get("estimates")
        if not isinstance(estimates, list) or not estimates:
            return None
        current = next((item for item in estimates if isinstance(item, Mapping)), None)
        if not isinstance(current, Mapping):
            return None
        return AnalystExpectationProfile(
            ticker=ticker.upper(),
            revenue_growth_estimate_current_q=self._as_float(current.get("estimatedRevenueGrowth")),
            eps_growth_estimate_current_q=self._as_float(current.get("estimatedEPSGrowth")),
            revenue_analyst_count=self._as_int(current.get("revenueAnalystCount")),
            eps_analyst_count=self._as_int(current.get("epsAnalystCount")),
        )

    def _get_next_earnings_event_finnhub_sync(self, ticker: str) -> EarningsEvent | None:
        rows = self._fetch_finnhub_json("calendar/earnings", symbol=ticker.upper())
        earnings_calendar = rows.get("earningsCalendar") if isinstance(rows, Mapping) else None
        if isinstance(earnings_calendar, Mapping):
            earnings_calendar = earnings_calendar.get("earningsCalendar")
        if not isinstance(earnings_calendar, list):
            return None
        now = datetime.now(timezone.utc)
        for item in earnings_calendar:
            if not isinstance(item, Mapping):
                continue
            earnings_at = self._parse_datetime(item.get("date"))
            if earnings_at is None or earnings_at < now:
                continue
            return EarningsEvent(
                ticker=ticker.upper(),
                earnings_at=earnings_at,
                eps_estimate=self._as_float(item.get("epsEstimate")),
                reported_eps=self._as_float(item.get("epsActual")),
                surprise_pct=self._as_float(item.get("surprisePercent")),
            )
        return None

    def _get_analyst_expectation_profile_finnhub_sync(self, ticker: str) -> AnalystExpectationProfile | None:
        payload = self._fetch_finnhub_json("stock/eps-estimate", symbol=ticker.upper())
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list) or not data:
            return None
        current = next((item for item in data if isinstance(item, Mapping)), None)
        if not isinstance(current, Mapping):
            return None
        return AnalystExpectationProfile(
            ticker=ticker.upper(),
            revenue_growth_estimate_current_q=self._as_float(current.get("revenueGrowth")),
            eps_growth_estimate_current_q=self._as_float(current.get("epsAvg")),
            revenue_analyst_count=self._as_int(current.get("numberAnalysts")),
            eps_analyst_count=self._as_int(current.get("numberAnalysts")),
        )

    def _get_analyst_rating_profile_sync(self, ticker: str) -> AnalystRatingsProfile | None:
        profile = self._get_analyst_rating_profile_finnhub_sync(ticker)
        if profile is not None:
            return profile
        return self._get_analyst_rating_profile_yfinance_sync(ticker)

    def _get_analyst_rating_profile_finnhub_sync(self, ticker: str) -> AnalystRatingsProfile | None:
        payload = self._fetch_finnhub_json("stock/recommendation", symbol=ticker.upper())
        rows = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list) and isinstance(payload, list):
            rows = payload
        if not isinstance(rows, list) or not rows:
            return None
        latest = next((item for item in rows if isinstance(item, Mapping)), None)
        if not isinstance(latest, Mapping):
            return None
        buy_count = (self._as_int(latest.get("strongBuy")) or 0) + (self._as_int(latest.get("buy")) or 0)
        hold_count = self._as_int(latest.get("hold"))
        sell_count = (self._as_int(latest.get("sell")) or 0) + (self._as_int(latest.get("strongSell")) or 0)
        target_price = self._get_rating_target_price_yfinance_sync(ticker)
        current_price = self._get_rating_reference_price_sync(ticker)
        upside_pct = self._safe_pct_change(target_price, current_price) if target_price is not None and current_price is not None else None
        return AnalystRatingsProfile(
            ticker=ticker.upper(),
            consensus_signal=self._classify_analyst_consensus_signal(buy_count=buy_count, hold_count=hold_count, sell_count=sell_count),
            buy_count=buy_count,
            hold_count=hold_count,
            sell_count=sell_count,
            target_price=target_price,
            upside_pct=upside_pct,
            source="finnhub",
        )

    def _get_analyst_rating_profile_yfinance_sync(self, ticker: str) -> AnalystRatingsProfile | None:
        ticker_client = yf.Ticker(ticker)
        try:
            info = ticker_client.info or {}
            fast_info = ticker_client.fast_info or {}
        except Exception as exc:
            logger.warning("Analyst ratings unavailable for {} via yfinance: {}", ticker, exc)
            return None
        if not isinstance(info, Mapping):
            info = {}
        if not isinstance(fast_info, Mapping):
            fast_info = {}
        target_price = self._as_float(info.get("targetMeanPrice")) or self._as_float(info.get("targetMedianPrice"))
        current_price = (
            self._as_float(info.get("currentPrice"))
            or self._as_float(info.get("regularMarketPrice"))
            or self._as_float(fast_info.get("lastPrice"))
        )
        recommendation_key = str(info.get("recommendationKey") or "").strip().casefold()
        analyst_count = self._as_int(info.get("numberOfAnalystOpinions"))
        if target_price is None and not recommendation_key and analyst_count is None:
            return None
        return AnalystRatingsProfile(
            ticker=ticker.upper(),
            consensus_signal=self._normalize_yfinance_recommendation_signal(recommendation_key),
            buy_count=None,
            hold_count=analyst_count,
            sell_count=None,
            target_price=target_price,
            upside_pct=self._safe_pct_change(target_price, current_price) if target_price is not None and current_price is not None else None,
            source="yfinance",
        )

    def _get_rating_target_price_yfinance_sync(self, ticker: str) -> float | None:
        ticker_client = yf.Ticker(ticker)
        try:
            info = ticker_client.info or {}
        except Exception:
            return None
        if not isinstance(info, Mapping):
            return None
        return self._as_float(info.get("targetMeanPrice")) or self._as_float(info.get("targetMedianPrice"))

    def _get_rating_reference_price_sync(self, ticker: str) -> float | None:
        try:
            quote = self._get_latest_price_yfinance_sync(ticker)
        except Exception:
            quote = None
        if quote is not None:
            return quote.price
        ticker_client = yf.Ticker(ticker)
        try:
            info = ticker_client.info or {}
            fast_info = ticker_client.fast_info or {}
        except Exception:
            return None
        if not isinstance(info, Mapping):
            info = {}
        if not isinstance(fast_info, Mapping):
            fast_info = {}
        return (
            self._as_float(info.get("currentPrice"))
            or self._as_float(info.get("regularMarketPrice"))
            or self._as_float(fast_info.get("lastPrice"))
        )

    @staticmethod
    def _classify_analyst_consensus_signal(
        *,
        buy_count: int | None,
        hold_count: int | None,
        sell_count: int | None,
    ) -> str | None:
        buys = int(buy_count or 0)
        holds = int(hold_count or 0)
        sells = int(sell_count or 0)
        if buys == 0 and holds == 0 and sells == 0:
            return None
        if buys >= max(holds, sells) + 2:
            return "bullish"
        if sells >= max(buys, holds) + 2:
            return "bearish"
        if holds >= max(buys, sells):
            return "neutral"
        return "mixed"

    @staticmethod
    def _normalize_yfinance_recommendation_signal(value: str) -> str | None:
        normalized = str(value or "").strip().casefold()
        if not normalized:
            return None
        if normalized in {"buy", "strong_buy", "outperform"}:
            return "bullish"
        if normalized in {"hold", "neutral"}:
            return "neutral"
        if normalized in {"underperform", "sell", "strong_sell"}:
            return "bearish"
        return normalized

    def _get_insider_transaction_summary_sync(self, ticker: str) -> InsiderTransactionSummary | None:
        ticker_upper = ticker.strip().upper()
        if not ticker_upper:
            return None
        mapping = self._load_sec_ticker_mapping()
        sec_identity = mapping.get(ticker_upper)
        cik = (sec_identity or {}).get("cik")
        if cik:
            submissions = self._fetch_sec_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
            filing_events = self._extract_recent_ownership_filing_events(ticker_upper, cik, submissions)
            if filing_events:
                summary = self._summarize_sec_insider_transactions(ticker_upper, filing_events)
                if summary is not None:
                    return summary
        insider_signal, _ = self._fetch_market_alternative_signals(ticker_upper)
        if insider_signal is None:
            return None
        return InsiderTransactionSummary(
            ticker=ticker_upper,
            signal=insider_signal,
            net_shares=None,
            net_value=None,
            buy_count=0,
            sell_count=0,
            transaction_count=0,
            last_filed_at=None,
            recent_transactions=(),
        )

    def _extract_recent_ownership_filing_events(
        self,
        ticker: str,
        cik: str | None,
        submissions: Mapping[str, Any] | None,
    ) -> tuple[FilingEvent, ...]:
        recent = ((submissions or {}).get("filings") or {}).get("recent")
        if not isinstance(recent, Mapping):
            return ()
        forms = list(recent.get("form") or [])
        filing_dates = list(recent.get("filingDate") or [])
        report_dates = list(recent.get("reportDate") or [])
        accession_numbers = list(recent.get("accessionNumber") or [])
        primary_documents = list(recent.get("primaryDocument") or [])
        items: list[FilingEvent] = []
        count = min(len(forms), len(filing_dates), len(accession_numbers), len(primary_documents))
        for index in range(count):
            form = str(forms[index] or "").strip().upper()
            if form not in {"3", "3/A", "4", "4/A", "5", "5/A"}:
                continue
            accession_number = str(accession_numbers[index] or "").strip()
            primary_document = str(primary_documents[index] or "").strip() or None
            filing_date = self._parse_datetime(filing_dates[index])
            report_date = self._parse_datetime(report_dates[index]) if index < len(report_dates) else None
            primary_url = None
            if cik and accession_number and primary_document:
                primary_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number.replace('-', '')}/{primary_document}"
            items.append(
                FilingEvent(
                    ticker=ticker,
                    cik=cik,
                    form=form,
                    filing_date=filing_date,
                    report_date=report_date,
                    accession_number=accession_number or None,
                    primary_document=primary_document,
                    primary_document_url=primary_url,
                )
            )
        items.sort(key=lambda item: item.filing_date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return tuple(items[:8])

    def _summarize_sec_insider_transactions(
        self,
        ticker: str,
        filings: Sequence[FilingEvent],
    ) -> InsiderTransactionSummary | None:
        transactions: list[InsiderTransaction] = []
        for filing in filings[:6]:
            if not filing.primary_document_url:
                continue
            raw_document = self._fetch_sec_raw_document_text(filing.primary_document_url)
            parsed = self._parse_sec_insider_transactions(raw_document, filing)
            if parsed:
                transactions.extend(parsed)
        if not transactions:
            return None
        ordered = sorted(
            transactions,
            key=lambda item: item.filed_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        net_shares = 0.0
        net_value = 0.0
        buy_count = 0
        sell_count = 0
        for item in ordered:
            shares = self._as_float(item.shares) or 0.0
            value = self._as_float(item.value)
            if value is None and shares and item.price_per_share is not None:
                value = shares * item.price_per_share
            if item.trade_type == "buy":
                buy_count += 1
                net_shares += shares
                net_value += value or 0.0
            elif item.trade_type == "sell":
                sell_count += 1
                net_shares -= shares
                net_value -= value or 0.0
        signal = self._classify_insider_summary_signal(net_shares=net_shares, buy_count=buy_count, sell_count=sell_count)
        last_filed_at = next((item.filed_at for item in ordered if item.filed_at is not None), None)
        return InsiderTransactionSummary(
            ticker=ticker,
            signal=signal,
            net_shares=round(net_shares, 2),
            net_value=round(net_value, 2),
            buy_count=buy_count,
            sell_count=sell_count,
            transaction_count=len(ordered),
            last_filed_at=last_filed_at,
            recent_transactions=tuple(ordered[:6]),
        )

    def _fetch_sec_raw_document_text(self, url: str) -> str | None:
        try:
            response = self._get_http_client().get(
                url,
                headers={"User-Agent": self.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.warning("SEC raw document fetch failed for {}: {}", url, exc)
            return None

    def _parse_sec_insider_transactions(
        self,
        raw_document: str | None,
        filing: FilingEvent,
    ) -> list[InsiderTransaction]:
        if not raw_document:
            return []
        text = str(raw_document).strip()
        if not text:
            return []
        xml_text = text
        if "<ownershipDocument" not in xml_text:
            match = re.search(r"(<ownershipDocument\b.*?</ownershipDocument>)", xml_text, re.I | re.S)
            if match:
                xml_text = match.group(1)
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        transactions: list[InsiderTransaction] = []
        namespaces = {"x": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        path_prefix = ".//x:" if namespaces else ".//"
        nodes = root.findall(f"{path_prefix}nonDerivativeTransaction", namespaces) + root.findall(
            f"{path_prefix}derivativeTransaction",
            namespaces,
        )
        fallback_date = self._parse_datetime(self._extract_xml_text(root, "periodOfReport", namespaces)) or filing.filing_date
        for node in nodes:
            code = str(
                self._extract_xml_text(node, "transactionCoding/transactionAcquiredDisposedCode/value", namespaces) or ""
            ).strip().upper()
            trade_type = "buy" if code == "A" else "sell" if code == "D" else code.casefold() or "other"
            shares = self._as_float(self._extract_xml_text(node, "transactionAmounts/transactionShares/value", namespaces))
            price_per_share = self._as_float(self._extract_xml_text(node, "transactionAmounts/transactionPricePerShare/value", namespaces))
            filed_at = self._parse_datetime(self._extract_xml_text(node, "transactionDate/value", namespaces)) or fallback_date
            value = shares * price_per_share if shares is not None and price_per_share is not None else None
            if shares is None and value is None:
                continue
            transactions.append(
                InsiderTransaction(
                    filed_at=filed_at,
                    trade_type=trade_type,
                    shares=shares,
                    price_per_share=price_per_share,
                    value=value,
                    source_url=filing.primary_document_url,
                )
            )
        return transactions

    @staticmethod
    def _extract_xml_text(node: ET.Element, path: str, namespaces: Mapping[str, str]) -> str | None:
        if not path:
            return None
        resolved_path = path
        if namespaces:
            resolved_path = "/".join(f"x:{segment}" for segment in path.split("/") if segment)
        element = node.find(resolved_path, namespaces)
        if element is None or element.text is None:
            return None
        text = str(element.text).strip()
        return text or None

    @staticmethod
    def _classify_insider_summary_signal(
        *,
        net_shares: float | None,
        buy_count: int,
        sell_count: int,
    ) -> str | None:
        if net_shares is None and buy_count == 0 and sell_count == 0:
            return None
        if (net_shares or 0.0) > 0 and buy_count >= sell_count:
            return "accumulating"
        if (net_shares or 0.0) < 0 and sell_count >= buy_count:
            return "selling"
        if buy_count or sell_count:
            return "mixed"
        return None

    def _get_corporate_actions_sync(self, ticker: str, lookback_days: int) -> list[CorporateActionEvent]:
        return self._get_corporate_actions_yfinance_sync(ticker, lookback_days=lookback_days)

    def _get_corporate_actions_yfinance_sync(self, ticker: str, *, lookback_days: int) -> list[CorporateActionEvent]:
        ticker_client = yf.Ticker(ticker)
        try:
            actions = ticker_client.actions
        except Exception as exc:
            logger.warning("Corporate actions unavailable for {} via yfinance: {}", ticker, exc)
            return []
        if actions is None or actions.empty:
            return []
        frame = actions.reset_index()
        timestamp_column = str(frame.columns[0])
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))
        events: list[CorporateActionEvent] = []
        for _, row in frame.iterrows():
            ex_date = self._normalize_timestamp(row.get(timestamp_column))
            if ex_date is None or ex_date < cutoff:
                continue
            dividend = self._as_float(row.get("Dividends"))
            split_ratio = self._as_float(row.get("Stock Splits"))
            capital_gain = self._as_float(row.get("Capital Gains"))
            if dividend not in {None, 0.0}:
                events.append(
                    CorporateActionEvent(
                        ticker=ticker.upper(),
                        action_type="dividend",
                        ex_date=ex_date,
                        cash_amount=dividend,
                        source="yfinance",
                    )
                )
            if split_ratio not in {None, 0.0, 1.0}:
                events.append(
                    CorporateActionEvent(
                        ticker=ticker.upper(),
                        action_type="split",
                        ex_date=ex_date,
                        ratio=split_ratio,
                        source="yfinance",
                    )
                )
            if capital_gain not in {None, 0.0}:
                events.append(
                    CorporateActionEvent(
                        ticker=ticker.upper(),
                        action_type="capital_gain",
                        ex_date=ex_date,
                        cash_amount=capital_gain,
                        source="yfinance",
                    )
                )
        events.sort(key=lambda item: item.ex_date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return events

    def _get_etf_exposure_profile_sync(self, ticker: str) -> ETFExposureProfile | None:
        ticker_client = yf.Ticker(ticker)
        try:
            info = ticker_client.info or {}
        except Exception as exc:
            logger.warning("ETF exposure info unavailable for {}: {}", ticker, exc)
            info = {}
        if not isinstance(info, Mapping):
            info = {}
        top_holdings = self._extract_weighted_pairs(
            getattr(ticker_client, "fund_top_holdings", None),
            name_candidates=("name", "holding", "symbol", "asset"),
            weight_candidates=("holding percent", "holding_pct", "holdingpercent", "weight", "weight_pct", "% assets"),
        )
        sector_exposures = self._extract_weighted_pairs(
            getattr(ticker_client, "fund_sector_weightings", None),
            name_candidates=("sector", "name", "label"),
            weight_candidates=("weight", "weight_pct", "percentage", "value"),
        )
        country_exposures = self._extract_weighted_pairs(
            getattr(ticker_client, "fund_country_weightings", None),
            name_candidates=("country", "name", "label"),
            weight_candidates=("weight", "weight_pct", "percentage", "value"),
        )
        total_assets = self._as_float(info.get("totalAssets")) or self._as_float(info.get("netAssets"))
        fund_flow_1m_pct = self._as_float(
            info.get("fundFlow1M")
            or info.get("fundFlow1m")
            or info.get("fundInflowPct1M")
            or info.get("fundFlowPct1M")
        )
        concentration_score = round(
            sum(weight for _, weight in top_holdings[:5] if weight is not None),
            2,
        ) if top_holdings else None
        if not any(
            (
                self._as_optional_str(info.get("fundFamily")),
                self._as_optional_str(info.get("category")),
                total_assets,
                top_holdings,
                sector_exposures,
                country_exposures,
            )
        ):
            return None
        return ETFExposureProfile(
            ticker=ticker.upper(),
            fund_family=self._as_optional_str(info.get("fundFamily")),
            category=self._as_optional_str(info.get("category")),
            total_assets=total_assets,
            fund_flow_1m_pct=fund_flow_1m_pct,
            top_holdings=tuple(top_holdings[:5]),
            sector_exposures=tuple(sector_exposures[:6]),
            country_exposures=tuple(country_exposures[:6]),
            concentration_score=concentration_score,
            exposure_signal=self._classify_etf_exposure_signal(top_holdings, sector_exposures, country_exposures),
            source="yfinance",
        )

    def _extract_weighted_pairs(
        self,
        payload: Any,
        *,
        name_candidates: Sequence[str],
        weight_candidates: Sequence[str],
    ) -> list[tuple[str, float | None]]:
        if payload is None:
            return []
        rows: list[tuple[str, float | None]] = []
        normalized_name_keys = {item.casefold() for item in name_candidates}
        normalized_weight_keys = {item.casefold() for item in weight_candidates}
        if isinstance(payload, pd.DataFrame):
            frame = payload.reset_index()
            columns = {str(column).casefold(): str(column) for column in frame.columns}
            name_column = next((columns[key] for key in normalized_name_keys if key in columns), None)
            weight_column = next((columns[key] for key in normalized_weight_keys if key in columns), None)
            if name_column is None and len(frame.columns) >= 1:
                name_column = str(frame.columns[0])
            if weight_column is None and len(frame.columns) >= 2:
                weight_column = str(frame.columns[1])
            if name_column and weight_column:
                for _, row in frame.iterrows():
                    name = str(row.get(name_column) or "").strip()
                    weight = self._normalize_weight_pct(row.get(weight_column))
                    if name:
                        rows.append((name, weight))
        elif isinstance(payload, Mapping):
            for key, value in payload.items():
                name = str(key or "").strip()
                weight = self._normalize_weight_pct(value)
                if name:
                    rows.append((name, weight))
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    normalized_map = {str(key).casefold(): value for key, value in item.items()}
                    name = next((str(normalized_map[key]).strip() for key in normalized_name_keys if key in normalized_map), "")
                    weight = next((self._normalize_weight_pct(normalized_map[key]) for key in normalized_weight_keys if key in normalized_map), None)
                    if name:
                        rows.append((name, weight))
        deduped: dict[str, float | None] = {}
        for name, weight in rows:
            deduped[name] = weight
        ordered = sorted(
            deduped.items(),
            key=lambda item: (item[1] is None, -(item[1] or 0.0), item[0]),
        )
        return ordered

    def _normalize_weight_pct(self, value: Any) -> float | None:
        numeric = self._as_float(value)
        if numeric is None:
            return None
        if abs(numeric) <= 1.0:
            numeric *= 100.0
        return round(numeric, 2)

    @staticmethod
    def _classify_etf_exposure_signal(
        top_holdings: Sequence[tuple[str, float | None]],
        sector_exposures: Sequence[tuple[str, float | None]],
        country_exposures: Sequence[tuple[str, float | None]],
    ) -> str | None:
        top_holding_weight = next((weight for _, weight in top_holdings if weight is not None), None)
        sector_weight = next((weight for _, weight in sector_exposures if weight is not None), None)
        country_weight = next((weight for _, weight in country_exposures if weight is not None), None)
        concentration = sum(weight for _, weight in top_holdings[:5] if weight is not None)
        if top_holding_weight is not None and top_holding_weight >= 12.0:
            return "top_holding_concentrated"
        if sector_weight is not None and sector_weight >= 40.0:
            return "sector_concentrated"
        if country_weight is not None and country_weight >= 70.0:
            return "country_concentrated"
        if concentration >= 35.0:
            return "concentrated"
        if top_holdings or sector_exposures or country_exposures:
            return "diversified"
        return None

    def _get_company_intelligence_sync(self, ticker: str, company_name: str | None = None) -> CompanyIntelligence | None:
        ticker_upper = ticker.strip().upper()
        if not ticker_upper:
            return None
        mapping = self._load_sec_ticker_mapping()
        sec_identity = mapping.get(ticker_upper)
        cik = (sec_identity or {}).get("cik")
        resolved_name = company_name or (sec_identity or {}).get("title")
        submissions = self._fetch_sec_json(f"https://data.sec.gov/submissions/CIK{cik}.json") if cik else None
        companyfacts = self._fetch_sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json") if cik else None

        recent_filings = self._extract_recent_filing_events(ticker_upper, cik, submissions)
        revenue_rows = self._extract_company_fact_rows(
            companyfacts,
            (
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
            ),
        )
        ocf_rows = self._extract_company_fact_rows(
            companyfacts,
            (
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
                "NetCashProvidedByUsedInOperatingActivities",
            ),
        )
        capex_rows = self._extract_company_fact_rows(
            companyfacts,
            (
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "CapitalExpendituresIncurredButNotYetPaid",
                "PropertyPlantAndEquipmentAdditions",
            ),
        )
        debt_rows = self._extract_company_fact_rows(
            companyfacts,
            (
                "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebt",
                "LongTermDebtNoncurrent",
                "LongTermDebtCurrent",
            ),
        )
        share_rows = self._extract_company_fact_rows(
            companyfacts,
            (
                "CommonStockSharesOutstanding",
                "CommonStocksIncludingAdditionalPaidInCapital",
                "WeightedAverageNumberOfSharesOutstandingBasic",
            ),
        )
        one_off_rows = self._extract_company_fact_rows(
            companyfacts,
            (
                "RestructuringCharges",
                "AssetImpairmentCharges",
                "GainLossOnDispositionOfAssets1",
                "GainLossOnSaleOfAssetsNet",
            ),
        )

        revenue_latest, revenue_prior = self._latest_and_year_ago_values(revenue_rows)
        ocf_latest, _ = self._latest_and_year_ago_values(ocf_rows)
        capex_latest, _ = self._latest_and_year_ago_values(capex_rows)
        debt_latest, debt_prior = self._latest_and_year_ago_values(debt_rows)
        shares_latest, shares_prior = self._latest_and_year_ago_values(share_rows)
        one_off_latest, _ = self._latest_and_year_ago_values(one_off_rows)
        revenue_yoy_pct = self._safe_pct_change(revenue_latest, revenue_prior)
        debt_delta_pct = self._safe_pct_change(debt_latest, debt_prior)
        share_dilution_yoy_pct = self._safe_pct_change(shares_latest, shares_prior)
        free_cash_flow_latest = None
        if ocf_latest is not None:
            free_cash_flow_latest = ocf_latest - abs(capex_latest or 0.0)

        one_off_signal = None
        if one_off_latest is not None and revenue_latest not in {None, 0}:
            one_off_ratio = abs(one_off_latest) / abs(revenue_latest)
            if one_off_ratio >= 0.05:
                one_off_signal = "high"
            elif one_off_ratio >= 0.02:
                one_off_signal = "moderate"
            else:
                one_off_signal = "low"

        guidance_signal = self._infer_guidance_signal_from_filings(recent_filings)
        fallback_insider_signal, sentiment_signal = self._fetch_market_alternative_signals(ticker_upper)
        insider_summary = self._get_insider_transaction_summary_sync(ticker_upper)
        insider_signal = (insider_summary.signal if insider_summary is not None else None) or fallback_insider_signal
        analyst_profile = self._get_analyst_rating_profile_sync(ticker_upper)
        expectation_signal = self._infer_expectation_signal(ticker_upper)
        corporate_actions = tuple(self._get_corporate_actions_sync(ticker_upper, lookback_days=365)[:4])
        corporate_action_signal = self._classify_corporate_action_signal(corporate_actions)
        filing_highlights = self._build_company_filing_highlights(
            revenue_yoy_pct=revenue_yoy_pct,
            debt_delta_pct=debt_delta_pct,
            share_dilution_yoy_pct=share_dilution_yoy_pct,
            one_off_signal=one_off_signal,
            guidance_signal=guidance_signal,
            insider_signal=insider_signal,
            sentiment_signal=sentiment_signal,
            analyst_rating_signal=analyst_profile.consensus_signal if analyst_profile is not None else None,
            analyst_upside_pct=analyst_profile.upside_pct if analyst_profile is not None else None,
            insider_net_value=insider_summary.net_value if insider_summary is not None else None,
            corporate_action_signal=corporate_action_signal,
        )

        latest_10k = next((item.filing_date for item in recent_filings if item.form.startswith("10-K")), None)
        latest_10q = next((item.filing_date for item in recent_filings if item.form.startswith("10-Q")), None)
        latest_8k = next((item.filing_date for item in recent_filings if item.form.startswith("8-K")), None)

        return CompanyIntelligence(
            ticker=ticker_upper,
            company_name=resolved_name,
            cik=cik,
            latest_10k_filed_at=latest_10k,
            latest_10q_filed_at=latest_10q,
            latest_8k_filed_at=latest_8k,
            revenue_latest=revenue_latest,
            revenue_yoy_pct=revenue_yoy_pct,
            operating_cash_flow_latest=ocf_latest,
            free_cash_flow_latest=free_cash_flow_latest,
            debt_latest=debt_latest,
            debt_delta_pct=debt_delta_pct,
            share_dilution_yoy_pct=share_dilution_yoy_pct,
            one_off_signal=one_off_signal,
            guidance_signal=guidance_signal,
            insider_signal=insider_signal,
            sentiment_signal=sentiment_signal,
            earnings_expectation_signal=expectation_signal,
            analyst_rating_signal=analyst_profile.consensus_signal if analyst_profile is not None else None,
            analyst_buy_count=analyst_profile.buy_count if analyst_profile is not None else None,
            analyst_hold_count=analyst_profile.hold_count if analyst_profile is not None else None,
            analyst_sell_count=analyst_profile.sell_count if analyst_profile is not None else None,
            analyst_upside_pct=analyst_profile.upside_pct if analyst_profile is not None else None,
            insider_net_shares=insider_summary.net_shares if insider_summary is not None else None,
            insider_net_value=insider_summary.net_value if insider_summary is not None else None,
            insider_transaction_count=insider_summary.transaction_count if insider_summary is not None else None,
            insider_last_filed_at=insider_summary.last_filed_at if insider_summary is not None else None,
            corporate_action_signal=corporate_action_signal,
            recent_corporate_actions=corporate_actions,
            filing_highlights=filing_highlights,
            recent_filings=recent_filings[:6],
        )

    def _load_sec_ticker_mapping(self) -> dict[str, dict[str, str]]:
        cache_key = "sec_company_tickers"
        with self._cache_lock:
            cached = self._sec_ticker_cache.get(cache_key)
            if cached is not None:
                return dict(cached)
        payload = self._fetch_sec_json("https://www.sec.gov/files/company_tickers.json")
        mapping: dict[str, dict[str, str]] = {}
        if isinstance(payload, Mapping):
            for item in payload.values():
                if not isinstance(item, Mapping):
                    continue
                ticker = str(item.get("ticker") or "").strip().upper()
                cik = self._normalize_cik(item.get("cik_str"))
                title = str(item.get("title") or "").strip()
                if ticker and cik:
                    mapping[ticker] = {"cik": cik, "title": title}
        with self._cache_lock:
            self._sec_ticker_cache[cache_key] = dict(mapping)
        return mapping

    def _fetch_sec_json(self, url: str) -> dict[str, Any] | None:
        try:
            response = self._get_http_client().get(
                url,
                headers={"User-Agent": self.sec_user_agent, "Accept-Encoding": "gzip, deflate", "Accept": "application/json"},
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("SEC request failed for {}: {}", url, exc)
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def _extract_recent_filing_events(
        self,
        ticker: str,
        cik: str | None,
        submissions: Mapping[str, Any] | None,
    ) -> tuple[FilingEvent, ...]:
        recent = ((submissions or {}).get("filings") or {}).get("recent")
        if not isinstance(recent, Mapping):
            return ()
        forms = list(recent.get("form") or [])
        filing_dates = list(recent.get("filingDate") or [])
        report_dates = list(recent.get("reportDate") or [])
        accession_numbers = list(recent.get("accessionNumber") or [])
        primary_documents = list(recent.get("primaryDocument") or [])
        items: list[FilingEvent] = []
        count = min(len(forms), len(filing_dates), len(accession_numbers), len(primary_documents))
        for index in range(count):
            form = str(forms[index] or "").strip().upper()
            if form not in {"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A"}:
                continue
            accession_number = str(accession_numbers[index] or "").strip()
            primary_document = str(primary_documents[index] or "").strip() or None
            filing_date = self._parse_datetime(filing_dates[index])
            report_date = self._parse_datetime(report_dates[index]) if index < len(report_dates) else None
            primary_url = None
            if cik and accession_number and primary_document:
                primary_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number.replace('-', '')}/{primary_document}"
                )
            items.append(
                FilingEvent(
                    ticker=ticker,
                    cik=cik,
                    form=form,
                    filing_date=filing_date,
                    report_date=report_date,
                    accession_number=accession_number or None,
                    primary_document=primary_document,
                    primary_document_url=primary_url,
                )
            )
        items.sort(key=lambda item: item.filing_date or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return tuple(items)

    def _extract_company_fact_rows(
        self,
        payload: Mapping[str, Any] | None,
        concepts: Sequence[str],
    ) -> list[dict[str, Any]]:
        facts = (((payload or {}).get("facts") or {}).get("us-gaap") or {})
        if not isinstance(facts, Mapping):
            return []
        rows: list[dict[str, Any]] = []
        for concept in concepts:
            concept_payload = facts.get(concept)
            units = concept_payload.get("units") if isinstance(concept_payload, Mapping) else None
            if not isinstance(units, Mapping):
                continue
            for unit_rows in units.values():
                if not isinstance(unit_rows, list):
                    continue
                for item in unit_rows:
                    if not isinstance(item, Mapping):
                        continue
                    form = str(item.get("form") or "").strip().upper()
                    if form and form not in {"10-K", "10-K/A", "10-Q", "10-Q/A"}:
                        continue
                    end_date = self._parse_datetime(item.get("end"))
                    filed_at = self._parse_datetime(item.get("filed"))
                    value = self._as_float(item.get("val"))
                    if end_date is None or value is None:
                        continue
                    rows.append(
                        {
                            "concept": concept,
                            "form": form,
                            "end": end_date,
                            "filed": filed_at,
                            "value": value,
                        }
                    )
        deduped: dict[tuple[datetime, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["end"], str(row["form"]))
            current = deduped.get(key)
            if current is None or (row["filed"] or datetime.min.replace(tzinfo=timezone.utc)) > (current["filed"] or datetime.min.replace(tzinfo=timezone.utc)):
                deduped[key] = row
        result = list(deduped.values())
        result.sort(key=lambda item: (item["end"], item["filed"] or item["end"]), reverse=True)
        return result

    def _infer_guidance_signal_from_filings(self, filings: Sequence[FilingEvent]) -> str | None:
        positive = 0
        negative = 0
        stable = 0
        pattern_positive = re.compile(r"\b(raise[sd]?|increase[sd]?|improv(?:e|ed|ing)|stronger)\b.{0,40}\b(guidance|outlook|forecast)\b", re.I)
        pattern_negative = re.compile(r"\b(cut|cuts|cutting|lower(?:ed|s|ing)?|reduce[sd]?|weaker)\b.{0,40}\b(guidance|outlook|forecast)\b", re.I)
        pattern_stable = re.compile(r"\b(reaffirm(?:ed|s|ing)?|maintain(?:ed|s|ing)?|unchanged|reiterate[sd]?)\b.{0,50}\b(guidance|outlook|forecast)\b", re.I)
        for filing in filings[:3]:
            if not filing.primary_document_url:
                continue
            document_texts: list[str] = []
            primary_text = self._fetch_sec_document_text(filing.primary_document_url)
            if primary_text:
                document_texts.append(primary_text)
                if filing.form.startswith("8-K"):
                    document_texts.extend(
                        self._fetch_sec_exhibit_texts(
                            filing.primary_document_url,
                            primary_text,
                        )
                    )
            for text in document_texts[:3]:
                positive += len(pattern_positive.findall(text))
                negative += len(pattern_negative.findall(text))
                stable += len(pattern_stable.findall(text))
        if positive > negative and positive > 0:
            return "positive"
        if negative > positive and negative > 0:
            return "negative"
        if positive and negative:
            return "mixed"
        if stable > 0:
            return "stable"
        return None

    def _fetch_sec_document_text(self, url: str) -> str | None:
        with self._cache_lock:
            if url in self._document_text_cache:
                return self._document_text_cache[url]
        try:
            response = self._get_http_client().get(
                url,
                headers={"User-Agent": self.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
                follow_redirects=True,
            )
            response.raise_for_status()
            text = self._normalize_document_text(response.text)
        except Exception as exc:
            logger.warning("SEC filing document fetch failed for {}: {}", url, exc)
            text = None
        with self._cache_lock:
            self._document_text_cache[url] = text
        return text

    def _fetch_sec_exhibit_texts(self, filing_url: str, filing_text: str) -> list[str]:
        raw_document = ""
        try:
            response = self._get_http_client().get(
                filing_url,
                headers={"User-Agent": self.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
                follow_redirects=True,
            )
            response.raise_for_status()
            raw_document = response.text
        except Exception:
            raw_document = filing_text
        candidate_urls = self._extract_guidance_related_sec_links(filing_url, raw_document)
        texts: list[str] = []
        for url in candidate_urls[:2]:
            text = self._fetch_sec_document_text(url)
            if text:
                texts.append(text)
        return texts

    @staticmethod
    def _extract_guidance_related_sec_links(base_url: str, filing_text: str) -> list[str]:
        matches: list[str] = []
        anchor_pattern = re.compile(r"""<a[^>]+href=["'](?P<href>[^"']+)["'][^>]*>(?P<label>.*?)</a>""", re.I | re.S)
        for match in anchor_pattern.finditer(filing_text or ""):
            label = MarketDataClient._normalize_document_text(match.group("label"))
            context = f"{label} {match.group('href')}".casefold()
            if not any(
                token in context
                for token in ("99.1", "99-1", "ex-99", "earnings release", "press release", "guidance", "outlook", "transcript")
            ):
                continue
            resolved = urljoin(base_url, match.group("href"))
            if resolved not in matches:
                matches.append(resolved)
        return matches

    @staticmethod
    def _normalize_document_text(text: str) -> str:
        cleaned = unescape(text or "")
        cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", cleaned)
        cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
        cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
        cleaned = re.sub(r"(?is)</p\s*>", "\n", cleaned)
        cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
        cleaned = re.sub(r"\n\s+", "\n", cleaned)
        cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
        return cleaned[:250_000].strip()

    def _fetch_market_alternative_signals(self, ticker: str) -> tuple[str | None, str | None]:
        insider_signal = None
        sentiment_signal = None

        finnhub_recommendations = self._fetch_finnhub_json("stock/recommendation", symbol=ticker)
        recommendation_rows = finnhub_recommendations.get("data") if isinstance(finnhub_recommendations, Mapping) else None
        if not isinstance(recommendation_rows, list) and isinstance(finnhub_recommendations, list):
            recommendation_rows = finnhub_recommendations
        if isinstance(recommendation_rows, list) and recommendation_rows:
            latest = next((item for item in recommendation_rows if isinstance(item, Mapping)), None)
            if isinstance(latest, Mapping):
                buys = (self._as_int(latest.get("strongBuy")) or 0) + (self._as_int(latest.get("buy")) or 0)
                sells = (self._as_int(latest.get("sell")) or 0) + (self._as_int(latest.get("strongSell")) or 0)
                if buys > sells:
                    sentiment_signal = "bullish"
                elif sells > buys:
                    sentiment_signal = "bearish"

        insider_payload = self._fetch_finnhub_json("stock/insider-sentiment", symbol=ticker)
        insider_rows = insider_payload.get("data") if isinstance(insider_payload, Mapping) else None
        if isinstance(insider_rows, list) and insider_rows:
            mspr_total = sum(self._as_float(item.get("mspr")) or 0.0 for item in insider_rows if isinstance(item, Mapping))
            if mspr_total > 0.1:
                insider_signal = "accumulating"
            elif mspr_total < -0.1:
                insider_signal = "selling"

        if sentiment_signal is None:
            alpha_news = self._fetch_alpha_vantage_json(function="NEWS_SENTIMENT", symbol=ticker)
            feed = alpha_news.get("feed") if isinstance(alpha_news, Mapping) else None
            scores = (
                [self._as_float(item.get("overall_sentiment_score")) for item in feed if isinstance(item, Mapping)]
                if isinstance(feed, list)
                else []
            )
            scores = [score for score in scores if score is not None]
            if scores:
                average_score = sum(scores) / len(scores)
                if average_score >= 0.15:
                    sentiment_signal = "bullish"
                elif average_score <= -0.15:
                    sentiment_signal = "bearish"
        finra_short_signal = self._fetch_finra_short_volume_signal(ticker)
        if sentiment_signal is None and finra_short_signal == "elevated_short_pressure":
            sentiment_signal = "bearish"
        return insider_signal, sentiment_signal

    def _infer_expectation_signal(self, ticker: str) -> str | None:
        profile = self._get_analyst_expectation_profile_sync(ticker)
        if profile is None:
            return None
        eps_growth = self._as_float(profile.eps_growth_estimate_current_q)
        revenue_growth = self._as_float(profile.revenue_growth_estimate_current_q)
        if (eps_growth is not None and eps_growth >= 0.08) or (revenue_growth is not None and revenue_growth >= 0.06):
            return "supportive"
        if (eps_growth is not None and eps_growth <= -0.05) or (revenue_growth is not None and revenue_growth <= -0.03):
            return "weak"
        return "mixed"

    @staticmethod
    def _build_company_filing_highlights(
        *,
        revenue_yoy_pct: float | None,
        debt_delta_pct: float | None,
        share_dilution_yoy_pct: float | None,
        one_off_signal: str | None,
        guidance_signal: str | None,
        insider_signal: str | None,
        sentiment_signal: str | None,
        analyst_rating_signal: str | None = None,
        analyst_upside_pct: float | None = None,
        insider_net_value: float | None = None,
        corporate_action_signal: str | None = None,
    ) -> tuple[str, ...]:
        highlights: list[str] = []
        if revenue_yoy_pct is not None:
            highlights.append(f"revenue_yoy={revenue_yoy_pct:+.2f}%")
        if debt_delta_pct is not None:
            highlights.append(f"debt_delta={debt_delta_pct:+.2f}%")
        if share_dilution_yoy_pct is not None:
            highlights.append(f"share_dilution={share_dilution_yoy_pct:+.2f}%")
        if one_off_signal:
            highlights.append(f"one_off={one_off_signal}")
        if guidance_signal:
            highlights.append(f"guidance={guidance_signal}")
        if insider_signal:
            highlights.append(f"insider={insider_signal}")
        if sentiment_signal:
            highlights.append(f"sentiment={sentiment_signal}")
        if analyst_rating_signal:
            highlights.append(f"analyst={analyst_rating_signal}")
        if analyst_upside_pct is not None:
            highlights.append(f"analyst_upside={analyst_upside_pct:+.2f}%")
        if insider_net_value is not None:
            highlights.append(f"insider_net_value={insider_net_value:+.0f}")
        if corporate_action_signal:
            highlights.append(f"corporate_actions={corporate_action_signal}")
        return tuple(highlights[:6])

    @staticmethod
    def _classify_corporate_action_signal(actions: Sequence[CorporateActionEvent]) -> str | None:
        if not actions:
            return None
        action_types = {item.action_type for item in actions if item.action_type}
        if "split" in action_types:
            return "split"
        if "dividend" in action_types or "capital_gain" in action_types:
            return "shareholder_return"
        return "eventful"

    @staticmethod
    def _latest_and_year_ago_values(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
        latest = MarketDataClient._as_float(rows[0].get("value")) if rows else None
        year_ago = (
            MarketDataClient._as_float(rows[4].get("value"))
            if len(rows) >= 5
            else MarketDataClient._as_float(rows[1].get("value"))
            if len(rows) >= 2
            else None
        )
        return latest, year_ago

    @staticmethod
    def _safe_pct_change(current: float | None, prior: float | None) -> float | None:
        if current is None or prior in {None, 0}:
            return None
        return round(((current - prior) / abs(prior)) * 100.0, 2)

    @staticmethod
    def _normalize_cik(value: object) -> str | None:
        digits = str(value or "").strip()
        return digits.zfill(10) if digits.isdigit() else None

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

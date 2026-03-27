from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

import httpx
from cachetools import TTLCache
from loguru import logger

from invest_advisor_bot.runtime_diagnostics import diagnostics


@dataclass(slots=True, frozen=True)
class OrderFlowSnapshot:
    symbol: str
    bullish_premium: float | None
    bearish_premium: float | None
    call_put_ratio: float | None
    unusual_count: int | None
    sweep_count: int | None
    opening_flow_ratio: float | None
    sentiment: str | None
    captured_at: datetime
    source: str | None = None


class OrderFlowClient:
    """Optional order-flow client for Cboe Trade Alert style feeds with flexible JSON parsing."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        api_key: str = "",
        base_url: str = "",
        timeout_seconds: float = 12.0,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.enabled = bool(enabled)
        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self._cache = TTLCache(maxsize=128, ttl=max(60, int(cache_ttl_seconds)))
        self._lock = RLock()
        self._http_client: httpx.Client | None = None
        self._warning: str | None = None
        self._last_fetch_at: str | None = None

    async def aclose(self) -> None:
        with self._lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            client.close()

    def enabled_and_configured(self) -> bool:
        return self.enabled and bool(self.api_key and self.base_url)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": True,
                "enabled": self.enabled,
                "configured": bool(self.api_key and self.base_url),
                "base_url": self.base_url or None,
                "timeout_seconds": self.timeout_seconds,
                "warning": self._warning,
                "last_fetch_at": self._last_fetch_at,
                "cache_entries": len(self._cache),
            }

    async def get_order_flow(self, symbols: Sequence[str]) -> dict[str, OrderFlowSnapshot]:
        normalized = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        if not normalized or not self.enabled_and_configured():
            return {}
        results = await asyncio.gather(
            *[self._get_symbol_order_flow(symbol) for symbol in normalized],
            return_exceptions=True,
        )
        payload: dict[str, OrderFlowSnapshot] = {}
        for symbol, result in zip(normalized, results, strict=False):
            if isinstance(result, OrderFlowSnapshot):
                payload[symbol] = result
        return payload

    async def _get_symbol_order_flow(self, symbol: str) -> OrderFlowSnapshot | None:
        with self._lock:
            cached = self._cache.get(symbol)
        if isinstance(cached, OrderFlowSnapshot):
            return cached
        result = await asyncio.to_thread(self._get_symbol_order_flow_sync, symbol)
        if result is not None:
            with self._lock:
                self._cache[symbol] = result
        return result

    def _get_symbol_order_flow_sync(self, symbol: str) -> OrderFlowSnapshot | None:
        started_at = datetime.now(timezone.utc)
        try:
            client = self._get_http_client()
            response = client.get(
                f"{self.base_url}/v1/flow/{symbol}",
                headers=self._headers(),
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            with self._lock:
                self._warning = f"order_flow_fetch_failed: {exc}"
            logger.warning("Order flow fetch failed for {}: {}", symbol, exc)
            diagnostics.record_provider_latency(
                service="order_flow_client",
                provider="cboe_trade_alert",
                operation="order_flow",
                latency_ms=(datetime.now(timezone.utc) - started_at).total_seconds() * 1000.0,
                success=False,
            )
            return None
        snapshot = self._parse_order_flow_payload(symbol=symbol, payload=payload)
        diagnostics.record_provider_latency(
            service="order_flow_client",
            provider="cboe_trade_alert",
            operation="order_flow",
            latency_ms=(datetime.now(timezone.utc) - started_at).total_seconds() * 1000.0,
            success=snapshot is not None,
        )
        if snapshot is not None:
            with self._lock:
                self._last_fetch_at = snapshot.captured_at.isoformat()
        return snapshot

    def _parse_order_flow_payload(self, *, symbol: str, payload: Any) -> OrderFlowSnapshot | None:
        if isinstance(payload, Mapping):
            data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
        else:
            data = None
        if not isinstance(data, Mapping):
            return None
        bullish_premium = self._coerce_float(
            data.get("bullish_premium") or data.get("call_premium") or data.get("premium_bought_calls")
        )
        bearish_premium = self._coerce_float(
            data.get("bearish_premium") or data.get("put_premium") or data.get("premium_bought_puts")
        )
        call_put_ratio = self._coerce_float(
            data.get("call_put_ratio") or data.get("callPutRatio") or data.get("premium_call_put_ratio")
        )
        unusual_count = self._coerce_int(
            data.get("unusual_count") or data.get("unusualTrades") or data.get("unusual_activity_count")
        )
        sweep_count = self._coerce_int(data.get("sweep_count") or data.get("sweepTrades") or data.get("sweeps"))
        opening_flow_ratio = self._coerce_float(
            data.get("opening_flow_ratio") or data.get("openingRatio") or data.get("opening_flow_pct")
        )
        sentiment = self._classify_sentiment(
            bullish_premium=bullish_premium,
            bearish_premium=bearish_premium,
            call_put_ratio=call_put_ratio,
        )
        if all(value is None for value in (bullish_premium, bearish_premium, call_put_ratio, unusual_count, sweep_count)):
            return None
        return OrderFlowSnapshot(
            symbol=symbol,
            bullish_premium=bullish_premium,
            bearish_premium=bearish_premium,
            call_put_ratio=call_put_ratio,
            unusual_count=unusual_count,
            sweep_count=sweep_count,
            opening_flow_ratio=opening_flow_ratio,
            sentiment=sentiment,
            captured_at=datetime.now(timezone.utc),
            source="cboe_trade_alert",
        )

    def _get_http_client(self) -> httpx.Client:
        with self._lock:
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.timeout_seconds)
            return self._http_client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
            "User-Agent": "InvestAdvisorBot/0.2",
            "Accept": "application/json",
        }

    @staticmethod
    def _classify_sentiment(
        *,
        bullish_premium: float | None,
        bearish_premium: float | None,
        call_put_ratio: float | None,
    ) -> str | None:
        if bullish_premium is not None and bearish_premium is not None:
            if bullish_premium > bearish_premium * 1.25:
                return "bullish"
            if bearish_premium > bullish_premium * 1.25:
                return "bearish"
        if call_put_ratio is not None:
            if call_put_ratio >= 1.3:
                return "bullish"
            if call_put_ratio <= 0.75:
                return "bearish"
            return "mixed"
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return None if value is None else int(float(value))
        except (TypeError, ValueError):
            return None

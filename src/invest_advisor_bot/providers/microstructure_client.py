from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

from cachetools import TTLCache
from loguru import logger

from invest_advisor_bot.runtime_diagnostics import diagnostics


@dataclass(slots=True, frozen=True)
class MicrostructureSnapshot:
    symbol: str
    dataset: str
    schema: str
    best_bid: float | None
    best_ask: float | None
    bid_size: float | None
    ask_size: float | None
    spread_bps: float | None
    imbalance: float | None
    last_price: float | None
    last_size: float | None
    sample_count: int
    captured_at: datetime


class MicrostructureClient:
    """Optional Databento-backed L2/L3 microstructure summary client."""

    def __init__(
        self,
        *,
        api_key: str = "",
        equities_dataset: str = "",
        options_dataset: str = "",
        equities_schema: str = "mbp-10",
        options_schema: str = "mbp-10",
        lookback_minutes: int = 5,
        cache_ttl_seconds: int = 120,
    ) -> None:
        self.api_key = api_key.strip()
        self.equities_dataset = equities_dataset.strip()
        self.options_dataset = options_dataset.strip()
        self.equities_schema = equities_schema.strip() or "mbp-10"
        self.options_schema = options_schema.strip() or "mbp-10"
        self.lookback_minutes = max(1, int(lookback_minutes))
        self._cache: TTLCache[tuple[str, str, str], MicrostructureSnapshot | None] = TTLCache(maxsize=128, ttl=max(30, cache_ttl_seconds))
        self._lock = RLock()
        self._warning: str | None = None

    def available(self) -> bool:
        return bool(self.api_key and (self.equities_dataset or self.options_dataset))

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available(),
            "equities_dataset": self.equities_dataset,
            "options_dataset": self.options_dataset,
            "equities_schema": self.equities_schema,
            "options_schema": self.options_schema,
            "lookback_minutes": self.lookback_minutes,
            "warning": self._warning,
            "cache_entries": len(self._cache),
        }

    async def get_equity_snapshot(self, symbol: str) -> MicrostructureSnapshot | None:
        return await self._get_snapshot(symbol=symbol, dataset=self.equities_dataset, schema=self.equities_schema)

    async def get_option_snapshot(self, symbol: str) -> MicrostructureSnapshot | None:
        return await self._get_snapshot(symbol=symbol, dataset=self.options_dataset, schema=self.options_schema)

    async def get_equity_snapshot_batch(self, symbols: Sequence[str], *, limit: int = 4) -> dict[str, MicrostructureSnapshot]:
        unique = []
        for symbol in symbols:
            normalized = symbol.strip().upper()
            if normalized and normalized not in unique:
                unique.append(normalized)
            if len(unique) >= max(1, limit):
                break
        tasks = {symbol: self.get_equity_snapshot(symbol) for symbol in unique}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, MicrostructureSnapshot] = {}
        for symbol, result in zip(tasks.keys(), results, strict=False):
            if isinstance(result, Exception) or result is None:
                continue
            payload[symbol] = result
        return payload

    async def _get_snapshot(self, *, symbol: str, dataset: str, schema: str) -> MicrostructureSnapshot | None:
        normalized = symbol.strip().upper()
        if not normalized or not dataset or not self.api_key:
            return None
        cache_key = (normalized, dataset, schema)
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        result = await asyncio.to_thread(self._get_snapshot_sync, normalized, dataset, schema)
        with self._lock:
            self._cache[cache_key] = result
        return result

    def _get_snapshot_sync(self, symbol: str, dataset: str, schema: str) -> MicrostructureSnapshot | None:
        if not self.available():
            self._warning = "microstructure client disabled or missing API key/dataset"
            return None
        databento = self._load_databento()
        if databento is None:
            return None
        started_at = self._monotonic()
        end_at = datetime.now(timezone.utc)
        start_at = end_at - timedelta(minutes=self.lookback_minutes)
        try:
            client = databento.Historical(self.api_key)
            data = client.timeseries.get_range(
                dataset=dataset,
                schema=schema,
                symbols=[symbol],
                start=start_at.isoformat(),
                end=end_at.isoformat(),
            )
            frame = data.to_df()
        except Exception as exc:
            diagnostics.record_provider_latency(
                service="microstructure_client",
                provider="databento",
                operation="depth_snapshot",
                latency_ms=(self._monotonic() - started_at) * 1000.0,
                success=False,
            )
            logger.warning("Databento snapshot failed for {}: {}", symbol, exc)
            self._warning = str(exc)
            return None
        diagnostics.record_provider_latency(
            service="microstructure_client",
            provider="databento",
            operation="depth_snapshot",
            latency_ms=(self._monotonic() - started_at) * 1000.0,
            success=not frame.empty,
        )
        if frame.empty:
            return None
        last_row = frame.iloc[-1]
        best_bid = self._pick_numeric(last_row, ("bid_px_00", "bid_px", "bid_price", "bpx"))
        best_ask = self._pick_numeric(last_row, ("ask_px_00", "ask_px", "ask_price", "apx"))
        bid_size = self._pick_numeric(last_row, ("bid_sz_00", "bid_sz", "bid_size", "bsz"))
        ask_size = self._pick_numeric(last_row, ("ask_sz_00", "ask_sz", "ask_size", "asz"))
        last_price = self._pick_numeric(last_row, ("price", "last_px", "trade_px"))
        last_size = self._pick_numeric(last_row, ("size", "last_sz", "trade_sz"))
        midpoint = None if best_bid is None or best_ask is None else (best_bid + best_ask) / 2.0
        spread_bps = None if midpoint in {None, 0} or best_bid is None or best_ask is None else round(((best_ask - best_bid) / midpoint) * 10000.0, 2)
        imbalance = None
        if bid_size is not None and ask_size is not None and (bid_size + ask_size) > 0:
            imbalance = round(bid_size / (bid_size + ask_size), 3)
        captured_at = self._coerce_datetime(getattr(last_row, "name", None)) or end_at
        return MicrostructureSnapshot(
            symbol=symbol,
            dataset=dataset,
            schema=schema,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            spread_bps=spread_bps,
            imbalance=imbalance,
            last_price=last_price,
            last_size=last_size,
            sample_count=int(len(frame)),
            captured_at=captured_at,
        )

    def _load_databento(self) -> Any | None:
        try:
            import databento as db
        except Exception as exc:
            self._warning = f"databento_unavailable: {exc}"
            return None
        return db

    @staticmethod
    def _pick_numeric(row: Any, candidates: Sequence[str]) -> float | None:
        for key in candidates:
            try:
                value = row[key]
            except Exception:
                value = None
            try:
                if value is None:
                    continue
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        text = str(value)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _monotonic() -> float:
        import time

        return time.perf_counter()

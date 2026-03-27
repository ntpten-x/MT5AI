from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

from loguru import logger

from invest_advisor_bot.runtime_diagnostics import diagnostics


@dataclass(slots=True, frozen=True)
class LiveMarketEvent:
    symbol: str
    dataset: str
    schema: str
    event_type: str
    price: float | None
    size: float | None
    bid: float | None
    ask: float | None
    spread_bps: float | None
    captured_at: datetime


class LiveMarketStreamClient:
    """Best-effort Databento live stream sampler for production telemetry."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        api_key: str = "",
        dataset: str = "",
        schema: str = "trades",
        max_events_per_poll: int = 25,
        sample_timeout_seconds: float = 3.0,
    ) -> None:
        self.enabled_flag = bool(enabled)
        self.api_key = api_key.strip()
        self.dataset = dataset.strip()
        self.schema = schema.strip() or "trades"
        self.max_events_per_poll = max(1, int(max_events_per_poll))
        self.sample_timeout_seconds = max(0.5, float(sample_timeout_seconds))
        self._lock = RLock()
        self._warning: str | None = None
        self._last_poll_at: str | None = None
        self._last_event_at: str | None = None
        self._last_event_count = 0
        self._last_symbols: tuple[str, ...] = ()

    def available(self) -> bool:
        return self.enabled_flag and bool(self.api_key and self.dataset)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.available(),
                "enabled": self.enabled_flag,
                "dataset": self.dataset,
                "schema": self.schema,
                "max_events_per_poll": self.max_events_per_poll,
                "sample_timeout_seconds": self.sample_timeout_seconds,
                "warning": self._warning,
                "last_poll_at": self._last_poll_at,
                "last_event_at": self._last_event_at,
                "last_event_count": self._last_event_count,
                "last_symbols": list(self._last_symbols),
            }

    async def sample_events(self, symbols: Sequence[str]) -> list[LiveMarketEvent]:
        return await asyncio.to_thread(self._sample_events_sync, list(symbols))

    def _sample_events_sync(self, symbols: Sequence[str]) -> list[LiveMarketEvent]:
        normalized_symbols = tuple(dict.fromkeys(str(item or "").strip().upper() for item in symbols if str(item or "").strip()))
        if not normalized_symbols:
            return []
        if not self.available():
            self._set_warning("live market stream disabled or missing Databento credentials")
            return []
        databento = self._load_databento()
        if databento is None:
            return []

        started_at = self._monotonic()
        try:
            live_cls = getattr(databento, "Live", None)
            if live_cls is None:
                raise RuntimeError("databento Live client unavailable")
            try:
                client = live_cls(key=self.api_key)
            except TypeError:
                client = live_cls(self.api_key)
            subscribe = getattr(client, "subscribe", None)
            if not callable(subscribe):
                raise RuntimeError("databento live subscribe unavailable")
            subscribe(
                dataset=self.dataset,
                schema=self.schema,
                symbols=list(normalized_symbols),
            )
            records = self._collect_records(client)
        except Exception as exc:
            diagnostics.record_provider_latency(
                service="live_market_stream_client",
                provider="databento_live",
                operation="sample_events",
                latency_ms=(self._monotonic() - started_at) * 1000.0,
                success=False,
            )
            logger.warning("Live market stream sampling failed: {}", exc)
            self._set_warning(str(exc))
            return []
        diagnostics.record_provider_latency(
            service="live_market_stream_client",
            provider="databento_live",
            operation="sample_events",
            latency_ms=(self._monotonic() - started_at) * 1000.0,
            success=bool(records),
        )
        now = datetime.now(timezone.utc)
        with self._lock:
            self._last_poll_at = now.isoformat()
            self._last_symbols = normalized_symbols
            self._last_event_count = len(records)
            self._last_event_at = records[-1].captured_at.isoformat() if records else self._last_event_at
            self._warning = None if records else self._warning
        return records

    def _collect_records(self, client: Any) -> list[LiveMarketEvent]:
        deadline = self._monotonic() + self.sample_timeout_seconds
        events: list[LiveMarketEvent] = []
        if hasattr(client, "__iter__"):
            iterator = iter(client)
            while self._monotonic() < deadline and len(events) < self.max_events_per_poll:
                try:
                    raw = next(iterator)
                except StopIteration:
                    break
                except Exception as exc:
                    if len(events) >= 1:
                        logger.debug("Live stream iterator stopped after partial sample: {}", exc)
                        break
                    raise
                event = self._coerce_event(raw)
                if event is not None:
                    events.append(event)
        else:
            next_record = getattr(client, "next_record", None)
            if not callable(next_record):
                raise RuntimeError("databento live client is not iterable")
            while self._monotonic() < deadline and len(events) < self.max_events_per_poll:
                raw = next_record(timeout=self.sample_timeout_seconds)
                if raw is None:
                    break
                event = self._coerce_event(raw)
                if event is not None:
                    events.append(event)
        stop_fn = getattr(client, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        return events

    def _coerce_event(self, raw: Any) -> LiveMarketEvent | None:
        symbol = self._extract_symbol(raw)
        if not symbol:
            return None
        bid = self._extract_float(raw, ("bid_px_00", "bid_px", "bid_price", "bid"))
        ask = self._extract_float(raw, ("ask_px_00", "ask_px", "ask_price", "ask"))
        midpoint = None if bid is None or ask is None else (bid + ask) / 2.0
        spread_bps = None
        if midpoint not in {None, 0} and bid is not None and ask is not None:
            spread_bps = round(((ask - bid) / midpoint) * 10000.0, 2)
        event = LiveMarketEvent(
            symbol=symbol,
            dataset=self.dataset,
            schema=self.schema,
            event_type=str(getattr(raw, "__class__", type("record", (), {})).__name__).strip() or "record",
            price=self._extract_float(raw, ("price", "trade_px", "last_px")),
            size=self._extract_float(raw, ("size", "trade_sz", "last_sz")),
            bid=bid,
            ask=ask,
            spread_bps=spread_bps,
            captured_at=self._extract_datetime(raw) or datetime.now(timezone.utc),
        )
        return event

    @staticmethod
    def _extract_symbol(raw: Any) -> str | None:
        for key in ("symbol", "raw_symbol", "stype_in_symbol"):
            value = getattr(raw, key, None)
            if value:
                return str(value).strip().upper()
            if isinstance(raw, Mapping) and raw.get(key):
                return str(raw.get(key)).strip().upper()
        return None

    @staticmethod
    def _extract_float(raw: Any, candidates: Sequence[str]) -> float | None:
        for key in candidates:
            value = getattr(raw, key, None)
            if value is None and isinstance(raw, Mapping):
                value = raw.get(key)
            try:
                if value is None:
                    continue
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_datetime(raw: Any) -> datetime | None:
        for key in ("ts_event", "captured_at", "timestamp"):
            value = getattr(raw, key, None)
            if value is None and isinstance(raw, Mapping):
                value = raw.get(key)
            if value is None:
                continue
            if isinstance(value, datetime):
                return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            text = str(value).strip()
            if not text:
                continue
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        return None

    def _load_databento(self) -> Any | None:
        try:
            import databento as db
        except Exception as exc:
            self._set_warning(f"databento_live_unavailable: {exc}")
            return None
        return db

    def _set_warning(self, warning: str) -> None:
        with self._lock:
            self._warning = warning

    @staticmethod
    def _monotonic() -> float:
        import time

        return time.perf_counter()

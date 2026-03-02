from __future__ import annotations

import importlib
import re
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config import Settings
from modules.timeframes import resolve_mt5_timeframe
from modules.types import HeartbeatState


def _to_dict(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {key: _to_dict(item) for key, item in value._asdict().items()}
    if hasattr(value, "__dict__"):
        return {key: _to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(item) for item in value]
    return value


class MT5Bridge:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._mt5 = None
        self._resolved_symbols: dict[str, str] = {}
        self._reverse_symbols: dict[str, str] = {}

    @property
    def module(self):
        if self._mt5 is None:
            try:
                self._mt5 = importlib.import_module("MetaTrader5")
            except ImportError as exc:
                raise RuntimeError(
                    "MetaTrader5 package is not installed. Run `pip install -e .` first."
                ) from exc
        return self._mt5

    def connect(self) -> bool:
        mt5 = self.module
        base_kwargs: dict[str, Any] = {"timeout": self.settings.mt5.connect_timeout_ms}
        initialize_attempts = []
        if self.settings.mt5.terminal_path:
            initialize_attempts.append({**base_kwargs, "path": self.settings.mt5.terminal_path})
        initialize_attempts.append(dict(base_kwargs))

        last_error: Any = None
        connected = False
        for initialize_kwargs in initialize_attempts:
            if mt5.initialize(**initialize_kwargs):
                connected = True
                break
            last_error = mt5.last_error()
            try:
                mt5.shutdown()
            except Exception:
                pass
            time.sleep(1.0)

        if not connected:
            raise RuntimeError(f"MT5 initialize failed: {last_error}")

        if self.settings.mt5.login:
            login_ok = mt5.login(
                login=self.settings.mt5.login,
                password=self.settings.mt5.password,
                server=self.settings.mt5.server,
            )
            if not login_ok:
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        return True

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()

    def ensure_connection(self) -> bool:
        terminal_info = self.module.terminal_info()
        if terminal_info is not None:
            return True
        return self.connect()

    def connected(self) -> bool:
        return self.module.terminal_info() is not None

    def _configured_symbols(self) -> list[str]:
        return [str(symbol).upper() for symbol in self.settings.trading.symbols]

    def _normalize_symbol_key(self, value: str) -> str:
        return str(value).strip().upper()

    def _normalize_symbol_compact(self, value: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", self._normalize_symbol_key(value))

    def _remember_symbol(self, canonical_symbol: str, broker_symbol: str) -> str:
        canonical = self._normalize_symbol_key(canonical_symbol)
        broker = str(broker_symbol).strip()
        self._resolved_symbols[canonical] = broker
        self._reverse_symbols[self._normalize_symbol_key(broker)] = canonical
        return broker

    def _candidate_score(self, requested_symbol: str, candidate_symbol: str) -> int:
        requested = self._normalize_symbol_key(requested_symbol)
        candidate = self._normalize_symbol_key(candidate_symbol)
        if not requested or not candidate:
            return -1
        if candidate == requested:
            return 500

        compact_requested = self._normalize_symbol_compact(requested)
        compact_candidate = self._normalize_symbol_compact(candidate)
        if compact_candidate == compact_requested:
            return 450

        if candidate.startswith(requested):
            suffix = candidate[len(requested) :]
            if suffix and len(suffix) <= 8 and re.fullmatch(r"[A-Z0-9._-]+", suffix):
                return 400 - len(suffix)

        if candidate.endswith(requested):
            prefix = candidate[: -len(requested)]
            if prefix and len(prefix) <= 8 and re.fullmatch(r"[A-Z0-9._-]+", prefix):
                return 360 - len(prefix)

        if compact_candidate.startswith(compact_requested):
            return 300 - max(0, len(compact_candidate) - len(compact_requested))

        if compact_candidate.endswith(compact_requested):
            return 280 - max(0, len(compact_candidate) - len(compact_requested))

        return -1

    def _candidate_sort_key(self, candidate: Any, requested_symbol: str) -> tuple[int, int, int, str]:
        name = str(getattr(candidate, "name", ""))
        visible = 1 if getattr(candidate, "visible", False) else 0
        trade_mode = int(getattr(candidate, "trade_mode", 0) or 0)
        score = self._candidate_score(requested_symbol, name)
        return (score, visible, trade_mode, name)

    def broker_symbol(self, symbol: str) -> str:
        requested = self._normalize_symbol_key(symbol)
        cached = self._resolved_symbols.get(requested)
        if cached:
            return cached

        mt5 = self.module
        configured_alias = self.settings.trading.symbol_aliases.get(requested)
        if configured_alias:
            return self._remember_symbol(requested, configured_alias)

        if mt5.symbol_info(requested) is not None:
            return self._remember_symbol(requested, requested)

        if not self.settings.trading.auto_discover_broker_symbols:
            return self._remember_symbol(requested, requested)

        candidates = []
        for candidate in mt5.symbols_get() or []:
            score = self._candidate_score(requested, getattr(candidate, "name", ""))
            if score < 0:
                continue
            candidates.append(candidate)

        if not candidates:
            return self._remember_symbol(requested, requested)

        selected = max(candidates, key=lambda candidate: self._candidate_sort_key(candidate, requested))
        return self._remember_symbol(requested, getattr(selected, "name", requested))

    def canonical_symbol(self, broker_symbol: str) -> str:
        broker_key = self._normalize_symbol_key(broker_symbol)
        cached = self._reverse_symbols.get(broker_key)
        if cached:
            return cached

        for canonical_symbol, resolved in self._resolved_symbols.items():
            if self._normalize_symbol_key(resolved) == broker_key:
                self._reverse_symbols[broker_key] = canonical_symbol
                return canonical_symbol

        configured_aliases = {
            self._normalize_symbol_key(canonical): self._normalize_symbol_key(alias)
            for canonical, alias in self.settings.trading.symbol_aliases.items()
        }
        for canonical_symbol, alias_symbol in configured_aliases.items():
            if alias_symbol == broker_key:
                self._reverse_symbols[broker_key] = canonical_symbol
                return canonical_symbol

        configured_symbols = self._configured_symbols()
        if broker_key in configured_symbols:
            self._reverse_symbols[broker_key] = broker_key
            return broker_key

        matches = [
            canonical_symbol
            for canonical_symbol in configured_symbols
            if self._candidate_score(canonical_symbol, broker_key) >= 0
        ]
        if matches:
            best = max(matches, key=lambda canonical_symbol: self._candidate_score(canonical_symbol, broker_key))
            self._remember_symbol(best, broker_symbol)
            return best
        return broker_key

    def _normalize_symbol_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {}
        symbol_value = payload.get("symbol") or payload.get("name")
        if not symbol_value:
            return payload
        broker_symbol = str(symbol_value)
        canonical_symbol = self.canonical_symbol(broker_symbol)
        normalized = dict(payload)
        normalized["symbol"] = canonical_symbol
        normalized["broker_symbol"] = broker_symbol
        return normalized

    def inspect_symbols(self, requested_symbols: list[str] | None = None) -> list[dict[str, Any]]:
        symbols = requested_symbols or list(self.settings.trading.symbols)
        mt5 = self.module
        catalogue = list(mt5.symbols_get() or [])
        results: list[dict[str, Any]] = []
        for requested_symbol in symbols:
            requested = self._normalize_symbol_key(requested_symbol)
            resolved = self.broker_symbol(requested)
            info = mt5.symbol_info(resolved)
            candidates = [
                str(getattr(item, "name", ""))
                for item in catalogue
                if self._candidate_score(requested, getattr(item, "name", "")) >= 0
            ]
            results.append(
                {
                    "requested_symbol": requested,
                    "broker_symbol": resolved,
                    "matched": info is not None,
                    "visible": bool(getattr(info, "visible", False)) if info is not None else False,
                    "path": getattr(info, "path", "") if info is not None else "",
                    "currency_base": getattr(info, "currency_base", "") if info is not None else "",
                    "currency_profit": getattr(info, "currency_profit", "") if info is not None else "",
                    "trade_mode": int(getattr(info, "trade_mode", 0) or 0) if info is not None else None,
                    "candidates": candidates[:10],
                }
            )
        return results

    def heartbeat(self) -> HeartbeatState:
        started = time.perf_counter()
        try:
            ok = self.ensure_connection()
            account = self.account_info() if ok else None
            message = "connected" if ok else "disconnected"
        except Exception as exc:
            ok = False
            account = None
            message = str(exc)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return HeartbeatState(
            ok=ok,
            message=message,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
            account=account,
        )

    def prepare_symbol(self, symbol: str) -> dict[str, Any]:
        mt5 = self.module
        broker_symbol = self.broker_symbol(symbol)
        if not mt5.symbol_select(broker_symbol, True):
            raise RuntimeError(f"Failed to select symbol {broker_symbol} for {symbol}: {mt5.last_error()}")
        info = mt5.symbol_info(broker_symbol)
        if info is None:
            raise RuntimeError(f"Symbol info unavailable for {broker_symbol} (requested {symbol}): {mt5.last_error()}")
        return self._normalize_symbol_payload(_to_dict(info))

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        broker_symbol = self.broker_symbol(symbol)
        info = self.module.symbol_info(broker_symbol)
        if info is None:
            raise RuntimeError(f"Symbol info unavailable for {broker_symbol} (requested {symbol}): {self.module.last_error()}")
        return self._normalize_symbol_payload(_to_dict(info))

    def symbol_tick(self, symbol: str) -> dict[str, Any]:
        broker_symbol = self.broker_symbol(symbol)
        tick = self.module.symbol_info_tick(broker_symbol)
        if tick is None:
            raise RuntimeError(f"Tick unavailable for {broker_symbol} (requested {symbol}): {self.module.last_error()}")
        payload = self._normalize_symbol_payload(_to_dict(tick))
        payload.setdefault("symbol", self.canonical_symbol(broker_symbol))
        payload.setdefault("broker_symbol", broker_symbol)
        return payload

    def filling_modes(self, symbol: str | None = None, symbol_info: dict[str, Any] | None = None) -> list[int]:
        mt5 = self.module
        info = symbol_info if symbol_info is not None else (self.symbol_info(symbol) if symbol else {})
        filling_flags = int(info.get("filling_mode", 0) or 0)
        trade_exemode = int(info.get("trade_exemode", -1) or -1)

        order_fok = int(getattr(mt5, "ORDER_FILLING_FOK", 0))
        order_ioc = int(getattr(mt5, "ORDER_FILLING_IOC", 1))
        order_return = int(getattr(mt5, "ORDER_FILLING_RETURN", 2))
        market_execution = int(getattr(mt5, "SYMBOL_TRADE_EXECUTION_MARKET", 2))

        candidates: list[int] = []
        if trade_exemode == market_execution:
            if filling_flags & 2:
                candidates.append(order_ioc)
            if filling_flags & 1:
                candidates.append(order_fok)
            if not candidates:
                candidates.append(order_ioc)
        else:
            candidates.append(order_return)
            if filling_flags & 2:
                candidates.append(order_ioc)
            if filling_flags & 1:
                candidates.append(order_fok)

        deduped: list[int] = []
        for mode in candidates:
            if mode not in deduped:
                deduped.append(mode)
        return deduped or [order_ioc]

    def account_info(self) -> dict[str, Any] | None:
        info = self.module.account_info()
        return _to_dict(info) if info is not None else None

    def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        broker_symbol = self.broker_symbol(symbol) if symbol else None
        positions = self.module.positions_get(symbol=broker_symbol) if broker_symbol else self.module.positions_get()
        return [self._normalize_symbol_payload(_to_dict(position)) for position in positions or []]

    def fetch_rates(self, symbol: str, timeframe: str, count: int, start_pos: int = 0) -> pd.DataFrame:
        mt5 = self.module
        broker_symbol = self.broker_symbol(symbol)
        rates = mt5.copy_rates_from_pos(
            broker_symbol,
            resolve_mt5_timeframe(mt5, timeframe),
            start_pos,
            count,
        )
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed for {broker_symbol} (requested {symbol}): {mt5.last_error()}")
        frame = pd.DataFrame(rates)
        if frame.empty:
            return frame
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame["symbol"] = self.canonical_symbol(broker_symbol)
        frame["broker_symbol"] = broker_symbol
        return frame

    def order_send(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        broker_symbol = None
        if payload.get("symbol"):
            broker_symbol = self.broker_symbol(str(payload["symbol"]))
            payload["symbol"] = broker_symbol
        result = self.module.order_send(payload)
        if result is None:
            raise RuntimeError(f"order_send failed: {self.module.last_error()}")
        response = _to_dict(result)
        if broker_symbol:
            response["broker_symbol"] = broker_symbol
            response["symbol"] = self.canonical_symbol(broker_symbol)
        return response

    def close_position(self, position: dict[str, Any], deviation: int, comment: str) -> dict[str, Any]:
        mt5 = self.module
        symbol = str(position["symbol"])
        broker_symbol = str(position.get("broker_symbol") or self.broker_symbol(symbol))
        tick = self.symbol_tick(symbol)
        side_type = position.get("type")
        price = tick["bid"] if side_type == mt5.POSITION_TYPE_BUY else tick["ask"]
        order_type = mt5.ORDER_TYPE_SELL if side_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "position": position["ticket"],
            "volume": position["volume"],
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": position.get("magic", 0),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_modes(symbol=symbol)[0],
        }
        return self.order_send(request)

    def modify_position(self, ticket: int, sl: float | None, tp: float | None, comment: str) -> dict[str, Any]:
        mt5 = self.module
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "sl": float(sl or 0.0),
            "tp": float(tp or 0.0),
            "comment": comment,
        }
        return self.order_send(request)

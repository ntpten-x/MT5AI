from __future__ import annotations

import argparse
import importlib
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from invest_advisor_bot.config import Settings, get_settings


GOLD_SYMBOL_KEYS = {"XAUUSD", "GOLD"}
SUCCESS_RETCODE_NAMES = ("TRADE_RETCODE_DONE", "TRADE_RETCODE_DONE_PARTIAL", "TRADE_RETCODE_PLACED")


@dataclass(slots=True, frozen=True)
class GoldSignal:
    side: str
    reason: str
    confidence: float
    price: float
    atr: float
    ema_fast: float
    ema_slow: float
    rsi: float


@dataclass(slots=True, frozen=True)
class OrderPlan:
    symbol: str
    broker_symbol: str
    side: str
    volume: float
    entry_price: float
    sl: float
    tp: float
    risk_amount: float
    dry_run: bool
    reason: str


def compact_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def assert_gold_symbol(symbol: str) -> None:
    compact = compact_symbol(symbol)
    if compact not in GOLD_SYMBOL_KEYS:
        raise ValueError(f"MT5 Gold Trader only allows XAUUSD/GOLD symbols, got {symbol!r}")


def timeframe_to_mt5(mt5: Any, value: str) -> Any:
    normalized = str(value or "M5").strip().upper()
    attr = f"TIMEFRAME_{normalized}"
    if not hasattr(mt5, attr):
        raise ValueError(f"Unsupported MT5 timeframe: {value}")
    return getattr(mt5, attr)


def build_indicator_frame(rates: pd.DataFrame, *, fast_ema: int, slow_ema: int, rsi_period: int, atr_period: int) -> pd.DataFrame:
    frame = rates.copy()
    if frame.empty:
        return frame
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ema_fast"] = frame["close"].ewm(span=fast_ema, adjust=False).mean()
    frame["ema_slow"] = frame["close"].ewm(span=slow_ema, adjust=False).mean()

    delta = frame["close"].diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = gains / losses.replace(0, np.nan)
    frame["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr"] = true_range.ewm(alpha=1 / atr_period, adjust=False).mean()
    return frame


def generate_gold_signal(frame: pd.DataFrame) -> GoldSignal:
    usable = frame.dropna(subset=["close", "ema_fast", "ema_slow", "rsi", "atr"])
    if usable.empty:
        return GoldSignal("flat", "insufficient_data", 0.0, 0.0, 0.0, 0.0, 0.0, 50.0)
    row = usable.iloc[-1]
    price = float(row["close"])
    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    rsi = float(row["rsi"])
    atr = float(row["atr"])
    if atr <= 0 or price <= 0:
        return GoldSignal("flat", "invalid_atr_or_price", 0.0, price, atr, ema_fast, ema_slow, rsi)

    ema_gap = abs(ema_fast - ema_slow) / max(price, 1e-9)
    confidence = float(np.clip(0.55 + (ema_gap * 250.0), 0.0, 0.95))
    if ema_fast > ema_slow and price >= ema_fast and 50.0 <= rsi <= 75.0:
        return GoldSignal("long", "ema_uptrend_rsi_confirmed", confidence, price, atr, ema_fast, ema_slow, rsi)
    if ema_fast < ema_slow and price <= ema_fast and 28.0 <= rsi <= 50.0:
        return GoldSignal("short", "ema_downtrend_rsi_confirmed", confidence, price, atr, ema_fast, ema_slow, rsi)
    return GoldSignal("flat", "no_gold_edge", confidence, price, atr, ema_fast, ema_slow, rsi)


def round_to_step(value: float, step: float, *, digits: int | None = None) -> float:
    if step <= 0:
        return round(value, digits) if digits is not None else value
    rounded = round(value / step) * step
    if digits is None:
        text = f"{step:.10f}".rstrip("0")
        digits = len(text.split(".")[1]) if "." in text else 0
    return round(rounded, digits)


def calculate_lot_size(
    *,
    equity: float,
    risk_fraction: float,
    entry_price: float,
    stop_price: float,
    symbol_info: Mapping[str, Any],
    max_lot: float,
) -> tuple[float, float]:
    risk_amount = max(0.0, float(equity) * float(risk_fraction))
    price_distance = abs(float(entry_price) - float(stop_price))
    if risk_amount <= 0 or price_distance <= 0:
        return 0.0, risk_amount

    tick_size = float(symbol_info.get("trade_tick_size") or symbol_info.get("point") or 0.0)
    tick_value = float(symbol_info.get("trade_tick_value") or 0.0)
    money_per_lot = (price_distance / tick_size) * tick_value if tick_size > 0 and tick_value > 0 else 0.0
    if money_per_lot <= 0:
        contract_size = float(symbol_info.get("trade_contract_size") or 100.0)
        money_per_lot = price_distance * contract_size
    if money_per_lot <= 0:
        return 0.0, risk_amount

    raw_volume = risk_amount / money_per_lot
    volume_min = float(symbol_info.get("volume_min") or 0.01)
    volume_step = float(symbol_info.get("volume_step") or 0.01)
    volume_max = min(float(symbol_info.get("volume_max") or max_lot), float(max_lot))
    if raw_volume + 1e-12 < volume_min:
        return 0.0, risk_amount
    volume = min(max(raw_volume, volume_min), volume_max)
    return round_to_step(volume, volume_step), risk_amount


class MT5GoldTrader:
    def __init__(self, settings: Settings, *, mt5_module: Any | None = None) -> None:
        self.settings = settings
        self._mt5 = mt5_module
        self._symbol_cache: dict[str, str] = {}
        self._lock = RLock()
        self._state = {
            "kill_switch": bool(settings.mt5_gold_kill_switch),
            "day": None,
            "day_start_equity": None,
            "peak_equity": None,
            "last_trade_at": None,
            "last_preflight_ok_at": None,
            "last_preflight": None,
            "audit": [],
        }
        self._load_state()
        if settings.mt5_gold_kill_switch:
            self._state["kill_switch"] = True
            self._persist_state()

    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            try:
                self._mt5 = importlib.import_module("MetaTrader5")
            except ImportError as exc:
                raise RuntimeError("MetaTrader5 package is not installed. Install with `pip install -e .[mt5]`.") from exc
        return self._mt5

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.settings.mt5_gold_enabled,
                "symbol": self.settings.mt5_gold_symbol,
                "timeframe": self.settings.mt5_gold_timeframe,
                "dry_run": self.settings.mt5_gold_dry_run,
                "allow_live": self.settings.mt5_gold_allow_live,
                "require_preflight_for_live": self.settings.mt5_gold_require_preflight_for_live,
                "kill_switch": bool(self._state.get("kill_switch")),
                "state_path": str(self.settings.mt5_gold_state_path),
                "last_trade_at": self._state.get("last_trade_at"),
                "last_preflight_ok_at": self._state.get("last_preflight_ok_at"),
                "last_preflight": self._state.get("last_preflight"),
                "last_audit": (self._state.get("audit") or [])[-1] if self._state.get("audit") else None,
            }

    def set_kill_switch(self, enabled: bool, *, reason: str = "manual") -> dict[str, Any]:
        with self._lock:
            self._state["kill_switch"] = bool(enabled)
            self._audit("kill_switch_changed", {"enabled": bool(enabled), "reason": reason})
            self._persist_state()
            return self.status()

    def connect(self) -> None:
        mt5 = self.mt5
        kwargs = {"timeout": int(self.settings.mt5_connect_timeout_ms)}
        if self.settings.mt5_terminal_path.strip():
            kwargs["path"] = self.settings.mt5_terminal_path.strip()
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        if int(self.settings.mt5_login or 0) > 0:
            if not mt5.login(
                login=int(self.settings.mt5_login),
                password=self.settings.mt5_password,
                server=self.settings.mt5_server,
            ):
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    def shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def run_forever(self) -> None:
        while True:
            try:
                result = self.run_once()
                print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                payload = {"status": "failed", "reason": str(exc), "at": datetime.now(timezone.utc).isoformat()}
                self._audit("cycle_failed", payload)
                self._persist_state()
                print(json.dumps(payload, ensure_ascii=False), flush=True)
            time.sleep(max(5, int(self.settings.mt5_gold_poll_seconds)))

    def run_once(self) -> dict[str, Any]:
        if not self.settings.mt5_gold_enabled:
            return {"status": "disabled", "reason": "MT5_GOLD_ENABLED=false"}
        assert_gold_symbol(self.settings.mt5_gold_symbol)
        self.connect()
        try:
            return self._run_once_connected()
        finally:
            self.shutdown()

    def preflight(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "status": "failed",
            "ok": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }

        def add(name: str, status: str, message: str, **context: Any) -> None:
            checks.append({"name": name, "status": status, "message": message, "context": context})

        try:
            assert_gold_symbol(self.settings.mt5_gold_symbol)
            add("symbol_scope", "pass", "Configured symbol is gold-only", symbol=self.settings.mt5_gold_symbol)
        except Exception as exc:
            add("symbol_scope", "fail", str(exc), symbol=self.settings.mt5_gold_symbol)
            self._record_preflight(summary)
            return summary

        if not self.settings.mt5_gold_enabled:
            add("enabled", "fail", "MT5_GOLD_ENABLED=false")
            self._record_preflight(summary)
            return summary
        add("enabled", "pass", "MT5 Gold Trader is enabled")

        if not self.settings.mt5_gold_dry_run and not self.settings.mt5_gold_allow_live:
            add("live_flags", "fail", "Live trading cannot run unless MT5_GOLD_ALLOW_LIVE=true")
        elif not self.settings.mt5_gold_dry_run and self.settings.mt5_gold_allow_live:
            add("live_flags", "warn", "Live trading flags are enabled")
        else:
            add("live_flags", "pass", "Dry-run mode is active")

        try:
            self.connect()
            add("mt5_connection", "pass", "MT5 initialized and login succeeded")
            broker_symbol = self._resolve_broker_symbol(self.settings.mt5_gold_symbol.strip().upper())
            symbol_info = self._symbol_info(broker_symbol)
            symbol_path = str(symbol_info.get("path") or "")
            if compact_symbol(broker_symbol) == "GOLD" and "METAL" not in compact_symbol(symbol_path):
                add("broker_symbol", "fail", "Resolved GOLD symbol is not in a metal path", broker_symbol=broker_symbol, path=symbol_path)
            else:
                add("broker_symbol", "pass", "Broker symbol resolved", broker_symbol=broker_symbol, path=symbol_path)
            if not self.mt5.symbol_select(broker_symbol, True):
                add("symbol_select", "fail", f"Could not select {broker_symbol}", last_error=str(self.mt5.last_error()))
            else:
                add("symbol_select", "pass", "Broker symbol selected", broker_symbol=broker_symbol)

            account = self._account_info()
            equity = float(account.get("equity") or account.get("balance") or 0.0)
            add("account", "pass" if equity > 0 else "fail", "Account equity snapshot", equity=equity, balance=account.get("balance"))

            spread = int(symbol_info.get("spread") or 0)
            spread_status = "pass" if spread <= int(self.settings.mt5_gold_max_spread_points) else "fail"
            add("spread", spread_status, "Spread check", spread=spread, max_spread=int(self.settings.mt5_gold_max_spread_points))

            positions = self._positions_for_symbol(broker_symbol)
            positions_status = "pass" if len(positions) < int(self.settings.mt5_gold_max_positions) else "fail"
            add("positions", positions_status, "Open position count", count=len(positions), max_positions=int(self.settings.mt5_gold_max_positions))

            rates = self._fetch_rates(broker_symbol)
            frame = build_indicator_frame(
                rates,
                fast_ema=int(self.settings.mt5_gold_fast_ema),
                slow_ema=int(self.settings.mt5_gold_slow_ema),
                rsi_period=int(self.settings.mt5_gold_rsi_period),
                atr_period=int(self.settings.mt5_gold_atr_period),
            )
            usable = frame.dropna(subset=["close", "ema_fast", "ema_slow", "rsi", "atr"])
            add(
                "market_data",
                "pass" if len(usable) >= 50 else "fail",
                "Rates and indicators are available",
                bars=len(rates),
                usable_bars=len(usable),
            )
            if not usable.empty:
                row = usable.iloc[-1]
                min_risk = self._minimum_lot_risk_amount(symbol_info, float(row["atr"]))
                risk_budget = equity * float(self.settings.mt5_gold_risk_per_trade_pct)
                status = "pass" if risk_budget >= min_risk else "warn"
                add(
                    "lot_risk_budget",
                    status,
                    "Risk budget compared with broker minimum lot",
                    equity=round(equity, 4),
                    risk_pct=float(self.settings.mt5_gold_risk_per_trade_pct),
                    risk_budget=round(risk_budget, 4),
                    estimated_min_lot_risk=round(min_risk, 4),
                    min_lot=float(symbol_info.get("volume_min") or 0.01),
                    max_lot=float(self.settings.mt5_gold_max_lot),
                )

            guard = self._evaluate_account_guard(account)
            if guard:
                add("account_guard", "fail", "Account guard halted trading", guard=guard)
            elif bool(self._state.get("kill_switch")):
                add("account_guard", "fail", "Kill switch is enabled")
            else:
                add("account_guard", "pass", "Account guard is clear")
        except Exception as exc:
            add("exception", "fail", str(exc))
        finally:
            self.shutdown()

        failed = [item for item in checks if item["status"] == "fail"]
        warned = [item for item in checks if item["status"] == "warn"]
        summary["ok"] = not failed
        summary["status"] = "pass" if not failed and not warned else ("warn" if not failed else "failed")
        self._record_preflight(summary)
        return summary

    def _run_once_connected(self) -> dict[str, Any]:
        symbol = self.settings.mt5_gold_symbol.strip().upper()
        broker_symbol = self._resolve_broker_symbol(symbol)
        symbol_info = self._symbol_info(broker_symbol)
        if not self.mt5.symbol_select(broker_symbol, True):
            raise RuntimeError(f"Failed to select MT5 symbol {broker_symbol}: {self.mt5.last_error()}")

        account = self._account_info()
        guard = self._evaluate_account_guard(account)
        if guard:
            return guard
        if bool(self._state.get("kill_switch")):
            return {"status": "halted", "reason": "kill_switch_enabled", "symbol": symbol, "broker_symbol": broker_symbol}

        spread = int(symbol_info.get("spread") or 0)
        if spread > int(self.settings.mt5_gold_max_spread_points):
            return self._skip("spread_too_wide", symbol, broker_symbol, {"spread": spread})

        positions = self._positions_for_symbol(broker_symbol)
        if len(positions) >= int(self.settings.mt5_gold_max_positions):
            return self._skip("max_positions_reached", symbol, broker_symbol, {"positions": len(positions)})

        interval_skip = self._min_interval_skip()
        if interval_skip:
            return self._skip("min_trade_interval_active", symbol, broker_symbol, interval_skip)

        rates = self._fetch_rates(broker_symbol)
        frame = build_indicator_frame(
            rates,
            fast_ema=int(self.settings.mt5_gold_fast_ema),
            slow_ema=int(self.settings.mt5_gold_slow_ema),
            rsi_period=int(self.settings.mt5_gold_rsi_period),
            atr_period=int(self.settings.mt5_gold_atr_period),
        )
        signal = generate_gold_signal(frame)
        if signal.side == "flat":
            return self._skip(signal.reason, symbol, broker_symbol, {"confidence": signal.confidence, "rsi": signal.rsi})

        tick = self._tick(broker_symbol)
        plan = self._build_order_plan(symbol, broker_symbol, signal, symbol_info, tick, account)
        if plan.volume <= 0:
            return self._skip("position_size_below_minimum", symbol, broker_symbol, {"risk_amount": plan.risk_amount})
        if plan.dry_run:
            self._record_trade_marker()
            self._audit("dry_run_order", plan.__dict__)
            self._persist_state()
            return {"status": "dry_run", "plan": plan.__dict__}
        if not self.settings.mt5_gold_allow_live:
            return {"status": "blocked", "reason": "MT5_GOLD_ALLOW_LIVE=false", "plan": plan.__dict__}
        preflight_guard = self._live_preflight_guard()
        if preflight_guard:
            return {"status": "blocked", "reason": preflight_guard, "plan": plan.__dict__}
        return self._send_order(plan)

    def _build_order_plan(
        self,
        symbol: str,
        broker_symbol: str,
        signal: GoldSignal,
        symbol_info: Mapping[str, Any],
        tick: Mapping[str, Any],
        account: Mapping[str, Any],
    ) -> OrderPlan:
        digits = int(symbol_info.get("digits") or 2)
        point = float(symbol_info.get("point") or 0.01)
        price_step = float(symbol_info.get("trade_tick_size") or point)
        entry = float(tick.get("ask") if signal.side == "long" else tick.get("bid"))
        sl_distance = max(point * 10.0, signal.atr * float(self.settings.mt5_gold_sl_atr_mult))
        tp_distance = max(point * 10.0, signal.atr * float(self.settings.mt5_gold_tp_atr_mult))
        if signal.side == "long":
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance
        sl = round_to_step(sl, price_step, digits=digits)
        tp = round_to_step(tp, price_step, digits=digits)
        volume, risk_amount = calculate_lot_size(
            equity=float(account.get("equity") or account.get("balance") or 0.0),
            risk_fraction=float(self.settings.mt5_gold_risk_per_trade_pct),
            entry_price=entry,
            stop_price=sl,
            symbol_info=symbol_info,
            max_lot=float(self.settings.mt5_gold_max_lot),
        )
        return OrderPlan(
            symbol=symbol,
            broker_symbol=broker_symbol,
            side=signal.side,
            volume=volume,
            entry_price=round_to_step(entry, price_step, digits=digits),
            sl=sl,
            tp=tp,
            risk_amount=risk_amount,
            dry_run=bool(self.settings.mt5_gold_dry_run),
            reason=signal.reason,
        )

    def _send_order(self, plan: OrderPlan) -> dict[str, Any]:
        mt5 = self.mt5
        order_type = mt5.ORDER_TYPE_BUY if plan.side == "long" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": plan.broker_symbol,
            "volume": plan.volume,
            "type": order_type,
            "price": plan.entry_price,
            "sl": plan.sl,
            "tp": plan.tp,
            "deviation": int(self.settings.mt5_gold_deviation_points),
            "magic": int(self.settings.mt5_gold_magic),
            "comment": f"{self.settings.mt5_gold_comment_prefix}:{plan.side}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(plan.broker_symbol),
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"MT5 order_send failed: {mt5.last_error()}")
        payload = self._to_dict(result)
        retcode = int(payload.get("retcode", -1))
        success_codes = {int(getattr(mt5, name)) for name in SUCCESS_RETCODE_NAMES if hasattr(mt5, name)}
        status = "filled" if retcode in success_codes else "rejected"
        if status == "filled":
            self._record_trade_marker()
        self._audit("live_order_result", {"status": status, "request": request, "response": payload})
        self._persist_state()
        return {"status": status, "retcode": retcode, "request": request, "response": payload}

    def _minimum_lot_risk_amount(self, symbol_info: Mapping[str, Any], atr: float) -> float:
        point = float(symbol_info.get("point") or 0.01)
        stop_distance = max(point * 10.0, float(atr) * float(self.settings.mt5_gold_sl_atr_mult))
        tick_size = float(symbol_info.get("trade_tick_size") or symbol_info.get("point") or 0.0)
        tick_value = float(symbol_info.get("trade_tick_value") or 0.0)
        volume_min = float(symbol_info.get("volume_min") or 0.01)
        money_per_lot = (stop_distance / tick_size) * tick_value if tick_size > 0 and tick_value > 0 else 0.0
        if money_per_lot <= 0:
            money_per_lot = stop_distance * float(symbol_info.get("trade_contract_size") or 100.0)
        return max(0.0, money_per_lot * volume_min)

    def _live_preflight_guard(self) -> str | None:
        if not self.settings.mt5_gold_require_preflight_for_live:
            return None
        raw = self._state.get("last_preflight_ok_at")
        if not raw:
            return "preflight_required"
        try:
            checked_at = datetime.fromisoformat(str(raw))
        except ValueError:
            return "preflight_required"
        max_age = timedelta(minutes=int(self.settings.mt5_gold_preflight_max_age_minutes))
        if datetime.now(timezone.utc) - checked_at > max_age:
            return "preflight_expired"
        return None

    def _resolve_broker_symbol(self, symbol: str) -> str:
        cached = self._symbol_cache.get(symbol)
        if cached:
            return cached
        mt5 = self.mt5
        if mt5.symbol_info(symbol) is not None:
            self._symbol_cache[symbol] = symbol
            return symbol
        candidates: list[tuple[int, str]] = []
        requested = compact_symbol(symbol)
        for item in mt5.symbols_get() or []:
            name = str(getattr(item, "name", ""))
            path = str(getattr(item, "path", ""))
            compact = compact_symbol(name)
            if compact == requested or compact.startswith(requested) or compact.endswith(requested):
                score = 500 if compact == requested else 300
                candidates.append((score, name))
                continue
            if requested == "XAUUSD" and compact == "GOLD" and "METAL" in compact_symbol(path):
                candidates.append((450, name))
        if not candidates:
            raise RuntimeError(f"No broker symbol found for {symbol}")
        selected = sorted(candidates, key=lambda item: (-item[0], len(item[1]), item[1]))[0][1]
        self._symbol_cache[symbol] = selected
        return selected

    def _symbol_info(self, broker_symbol: str) -> dict[str, Any]:
        info = self.mt5.symbol_info(broker_symbol)
        if info is None:
            raise RuntimeError(f"Symbol info unavailable for {broker_symbol}: {self.mt5.last_error()}")
        return self._to_dict(info)

    def _tick(self, broker_symbol: str) -> dict[str, Any]:
        tick = self.mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            raise RuntimeError(f"Tick unavailable for {broker_symbol}: {self.mt5.last_error()}")
        return self._to_dict(tick)

    def _account_info(self) -> dict[str, Any]:
        account = self.mt5.account_info()
        if account is None:
            raise RuntimeError(f"MT5 account_info unavailable: {self.mt5.last_error()}")
        return self._to_dict(account)

    def _fetch_rates(self, broker_symbol: str) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(
            broker_symbol,
            timeframe_to_mt5(self.mt5, self.settings.mt5_gold_timeframe),
            0,
            int(self.settings.mt5_gold_history_bars),
        )
        frame = pd.DataFrame(rates)
        if frame.empty:
            raise RuntimeError(f"No rates returned for {broker_symbol}")
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        return frame

    def _positions_for_symbol(self, broker_symbol: str) -> list[dict[str, Any]]:
        positions = self.mt5.positions_get(symbol=broker_symbol) or []
        return [self._to_dict(item) for item in positions]

    def _filling_mode(self, broker_symbol: str) -> int:
        mt5 = self.mt5
        info = self._symbol_info(broker_symbol)
        filling_flags = int(info.get("filling_mode", 0) or 0)
        if filling_flags & 2:
            return int(mt5.ORDER_FILLING_IOC)
        if filling_flags & 1:
            return int(mt5.ORDER_FILLING_FOK)
        return int(getattr(mt5, "ORDER_FILLING_RETURN", mt5.ORDER_FILLING_IOC))

    def _evaluate_account_guard(self, account: Mapping[str, Any]) -> dict[str, Any] | None:
        equity = float(account.get("equity") or account.get("balance") or 0.0)
        if equity <= 0:
            return {"status": "halted", "reason": "invalid_account_equity"}
        today = date.today().isoformat()
        with self._lock:
            if self._state.get("day") != today:
                self._state["day"] = today
                self._state["day_start_equity"] = equity
            peak = float(self._state.get("peak_equity") or equity)
            self._state["peak_equity"] = max(peak, equity)
            start = float(self._state.get("day_start_equity") or equity)
            daily_loss = max(0.0, (start - equity) / start) if start > 0 else 0.0
            drawdown = max(0.0, (float(self._state["peak_equity"]) - equity) / float(self._state["peak_equity"]))
            if daily_loss >= float(self.settings.mt5_gold_daily_loss_limit_pct):
                return self._trip_guard("daily_loss_limit", daily_loss, float(self.settings.mt5_gold_daily_loss_limit_pct))
            if drawdown >= float(self.settings.mt5_gold_max_drawdown_pct):
                return self._trip_guard("max_drawdown_limit", drawdown, float(self.settings.mt5_gold_max_drawdown_pct))
            self._persist_state()
        return None

    def _trip_guard(self, reason: str, value: float, threshold: float) -> dict[str, Any]:
        self._state["kill_switch"] = True
        payload = {"status": "halted", "reason": reason, "value": value, "threshold": threshold}
        self._audit("guard_tripped", payload)
        self._persist_state()
        return payload

    def _min_interval_skip(self) -> dict[str, Any] | None:
        raw = self._state.get("last_trade_at")
        if not raw:
            return None
        try:
            last = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        minimum = int(self.settings.mt5_gold_min_trade_interval_seconds)
        if elapsed < minimum:
            return {"elapsed_seconds": int(elapsed), "min_seconds": minimum}
        return None

    def _record_trade_marker(self) -> None:
        with self._lock:
            self._state["last_trade_at"] = datetime.now(timezone.utc).isoformat()
            self._persist_state()

    def _record_preflight(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            snapshot = dict(payload)
            self._state["last_preflight"] = snapshot
            if bool(snapshot.get("ok")):
                self._state["last_preflight_ok_at"] = str(snapshot.get("checked_at") or datetime.now(timezone.utc).isoformat())
            self._audit("preflight", {"status": snapshot.get("status"), "ok": snapshot.get("ok")})
            self._persist_state()

    def _skip(self, reason: str, symbol: str, broker_symbol: str, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {"status": "skipped", "reason": reason, "symbol": symbol, "broker_symbol": broker_symbol, "context": dict(context or {})}
        self._audit("cycle_skipped", payload)
        self._persist_state()
        return payload

    def _audit(self, event: str, payload: Mapping[str, Any]) -> None:
        rows = list(self._state.get("audit") or [])
        item = dict(payload)
        item["event"] = event
        item["at"] = datetime.now(timezone.utc).isoformat()
        rows.append(item)
        self._state["audit"] = rows[-500:]

    def _load_state(self) -> None:
        path = Path(self.settings.mt5_gold_state_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if isinstance(payload, dict):
            self._state.update(payload)

    def _persist_state(self) -> None:
        path = Path(self.settings.mt5_gold_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        if hasattr(value, "_asdict"):
            return dict(value._asdict())
        if hasattr(value, "__dict__"):
            return dict(vars(value))
        if isinstance(value, Mapping):
            return dict(value)
        return {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MT5 automated XAUUSD/GOLD trader")
    parser.add_argument("command", choices=["status", "preflight", "cycle", "run", "killswitch-on", "killswitch-off"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    trader = MT5GoldTrader(get_settings())
    if args.command == "status":
        print(json.dumps(trader.status(), ensure_ascii=False, indent=2, default=str), flush=True)
        return 0
    if args.command == "preflight":
        print(json.dumps(trader.preflight(), ensure_ascii=False, indent=2, default=str), flush=True)
        return 0
    if args.command == "killswitch-on":
        print(json.dumps(trader.set_kill_switch(True), ensure_ascii=False, indent=2, default=str), flush=True)
        return 0
    if args.command == "killswitch-off":
        print(json.dumps(trader.set_kill_switch(False), ensure_ascii=False, indent=2, default=str), flush=True)
        return 0
    if args.command == "cycle":
        print(json.dumps(trader.run_once(), ensure_ascii=False, indent=2, default=str), flush=True)
        return 0
    trader.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

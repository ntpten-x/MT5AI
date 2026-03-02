from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from config import Settings
from modules.execution import ExecutionEngine


class _Bridge:
    def __init__(self, positions: list[dict], symbol_info: dict):
        self._positions = positions
        self._symbol_info = symbol_info
        self.modified: list[dict] = []
        self.closed: list[dict] = []
        self.module = type("MT5", (), {"POSITION_TYPE_BUY": 0})()

    def positions_get(self, symbol: str | None = None):
        return list(self._positions)

    def symbol_info(self, symbol: str):
        return dict(self._symbol_info)

    def modify_position(self, ticket: int, sl: float | None, tp: float | None, comment: str):
        payload = {"ticket": ticket, "sl": sl, "tp": tp, "comment": comment}
        self.modified.append(payload)
        return payload

    def close_position(self, position: dict, deviation: int, comment: str):
        payload = {"ticket": position["ticket"], "comment": comment}
        self.closed.append(payload)
        return payload


class _RiskManager:
    @staticmethod
    def round_step(value: float, step: float, precision: int | None = None) -> float:
        if precision is None:
            precision = 2
        return round(value, precision)


class _Database:
    pass


class _Notifier:
    pass


def _engine(settings: Settings, bridge: _Bridge) -> ExecutionEngine:
    return ExecutionEngine(
        settings=settings,
        bridge=bridge,
        database=_Database(),
        risk_manager=_RiskManager(),
        notifier=_Notifier(),
    )


def test_manage_open_positions_trails_gold_when_profit_reaches_trigger():
    now = datetime(2026, 3, 2, 3, 15, tzinfo=timezone.utc)
    position = {
        "ticket": 1001,
        "type": 0,
        "price_open": 100.0,
        "price_current": 112.5,
        "sl": 90.0,
        "tp": 130.0,
        "time": int((now - timedelta(minutes=15)).timestamp()),
    }
    bridge = _Bridge(
        positions=[position],
        symbol_info={"digits": 2, "trade_tick_size": 0.01, "point": 0.01},
    )
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        risk={
            "auto_breakeven_enable": True,
            "symbol_auto_breakeven_rr": {"GOLD": 0.75},
            "symbol_trailing_trigger_rr": {"GOLD": 1.2},
            "symbol_trailing_lock_rr": {"GOLD": 0.35},
        },
    )
    engine = _engine(settings, bridge)

    frame = pd.DataFrame({"time": pd.date_range(end=now, periods=6, freq="5min", tz="UTC")})
    results = engine.manage_open_positions("GOLD", frame)

    assert bridge.modified
    assert bridge.modified[0]["sl"] == 103.5
    assert results[0]["action"] == "trail"


def test_manage_open_positions_time_exit_closes_stalled_gold_trade():
    now = datetime(2026, 3, 2, 3, 30, tzinfo=timezone.utc)
    position = {
        "ticket": 1002,
        "type": 0,
        "price_open": 100.0,
        "price_current": 101.0,
        "sl": 90.0,
        "tp": 130.0,
        "time": int((now - timedelta(minutes=45)).timestamp()),
    }
    bridge = _Bridge(
        positions=[position],
        symbol_info={"digits": 2, "trade_tick_size": 0.01, "point": 0.01},
    )
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        risk={
            "auto_breakeven_enable": False,
            "symbol_time_exit_bars": {"GOLD": 8},
            "symbol_time_exit_min_progress_rr": {"GOLD": 0.2},
        },
    )
    engine = _engine(settings, bridge)

    frame = pd.DataFrame({"time": pd.date_range(end=now, periods=10, freq="5min", tz="UTC")})
    results = engine.manage_open_positions("GOLD", frame)

    assert bridge.closed
    assert results[0]["action"] == "time_exit"

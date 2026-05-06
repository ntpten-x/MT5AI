from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from invest_advisor_bot.config import Settings
from invest_advisor_bot.mt5_gold_trader import (
    MT5GoldTrader,
    assert_gold_symbol,
    build_indicator_frame,
    calculate_lot_size,
    generate_gold_signal,
)


def _settings(tmp_path: Path, **overrides):
    values = {
        "MT5_GOLD_ENABLED": True,
        "MT5_GOLD_STATE_PATH": tmp_path / "mt5_gold_state.json",
        "MT5_GOLD_SYMBOL": "XAUUSD",
        "MT5_GOLD_DRY_RUN": True,
        "MT5_GOLD_ALLOW_LIVE": False,
        "MT5__LOGIN": 123,
        "MT5__PASSWORD": "secret",
        "MT5__SERVER": "demo",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_assert_gold_symbol_rejects_non_gold() -> None:
    assert_gold_symbol("XAUUSD")
    assert_gold_symbol("GOLD")
    with pytest.raises(ValueError, match="only allows"):
        assert_gold_symbol("EURUSD")


def test_generate_gold_signal_detects_uptrend() -> None:
    rows = []
    base = 2300.0
    for index in range(120):
        close = base + index * 0.35 + (0.8 if index % 5 else -0.9)
        rows.append(
            {
                "time": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "open": close - 0.4,
                "high": close + 1.2,
                "low": close - 1.0,
                "close": close,
                "tick_volume": 100,
            }
        )
    frame = build_indicator_frame(pd.DataFrame(rows), fast_ema=8, slow_ema=21, rsi_period=14, atr_period=14)

    signal = generate_gold_signal(frame)

    assert signal.side == "long"
    assert signal.reason == "ema_uptrend_rsi_confirmed"
    assert signal.confidence > 0.55


def test_calculate_lot_size_clamps_to_max_lot() -> None:
    volume, risk_amount = calculate_lot_size(
        equity=10_000,
        risk_fraction=0.01,
        entry_price=2300,
        stop_price=2290,
        symbol_info={
            "trade_tick_size": 0.01,
            "trade_tick_value": 1.0,
            "volume_min": 0.01,
            "volume_step": 0.01,
            "volume_max": 100.0,
        },
        max_lot=0.05,
    )

    assert risk_amount == 100
    assert volume == 0.05


def test_status_keeps_live_trading_disabled_by_default(tmp_path: Path) -> None:
    settings = _settings(tmp_path, MT5_GOLD_DRY_RUN=False, MT5_GOLD_ALLOW_LIVE=False)
    trader = MT5GoldTrader(settings)

    status = trader.status()

    assert status["dry_run"] is False
    assert status["allow_live"] is False


def test_live_preflight_guard_blocks_without_recent_preflight(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        MT5_GOLD_DRY_RUN=False,
        MT5_GOLD_ALLOW_LIVE=True,
        MT5_GOLD_REQUIRE_PREFLIGHT_FOR_LIVE=True,
    )
    trader = MT5GoldTrader(settings)

    assert trader._live_preflight_guard() == "preflight_required"

    trader._record_preflight({"ok": True, "status": "pass", "checked_at": datetime.now(timezone.utc).isoformat()})

    assert trader._live_preflight_guard() is None


def test_resolves_xauusd_to_broker_gold_spot_metal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fake_mt5 = SimpleNamespace(
        symbol_info=lambda symbol: None,
        symbols_get=lambda: [
            SimpleNamespace(name="BarrickGold", path="Stocks\\Canada\\BarrickGold"),
            SimpleNamespace(name="GOLD", path="Derivatives\\Spot Metals\\GOLD"),
        ],
    )
    trader = MT5GoldTrader(settings, mt5_module=fake_mt5)

    assert trader._resolve_broker_symbol("XAUUSD") == "GOLD"


def test_kill_switch_persists(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    trader = MT5GoldTrader(settings)

    trader.set_kill_switch(True, reason="test")
    reloaded = MT5GoldTrader(settings)

    assert reloaded.status()["kill_switch"] is True

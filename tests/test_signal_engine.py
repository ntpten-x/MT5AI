from __future__ import annotations

import pandas as pd

from config import Settings, TradingSettings
from modules.signals import SignalEngine


def test_signal_engine_long_signal():
    settings = Settings()
    engine = SignalEngine(settings)
    frame = pd.DataFrame(
        [
            {
                "close": 1.1234,
                "ema_fast": 1.1230,
                "ema_slow": 1.1220,
                "rsi_14": 58.0,
                "atr_14": 0.0012,
                "session_london": 1,
                "session_new_york": 0,
                "session_tokyo": 0,
                "session_overlap": 0,
            }
        ]
    )
    signal = engine.evaluate(
        "EURUSD",
        "M5",
        frame,
        long_probability=0.70,
        short_probability=0.20,
        chronos_direction=0.001,
    )
    assert signal.side == "long"


def test_signal_engine_crypto_all_sessions_skips_session_filter():
    settings = Settings(
        _env_file=None,
        trading=TradingSettings(
            symbols=["BTCUSD"],
            allowed_sessions=["london", "new_york"],
            symbol_allowed_sessions={"BTCUSD": ["ALL"]},
            symbol_allowed_hours_utc={"BTCUSD": list(range(24))},
        ),
    )
    engine = SignalEngine(settings)
    frame = pd.DataFrame(
        [
            {
                "time": pd.Timestamp("2026-03-01T23:00:00Z"),
                "close": 95000.0,
                "ema_fast": 95100.0,
                "ema_slow": 94800.0,
                "rsi_14": 58.0,
                "atr_14": 120.0,
                "session_london": 0,
                "session_new_york": 0,
                "session_tokyo": 0,
                "session_overlap": 0,
            }
        ]
    )
    signal = engine.evaluate(
        "BTCUSD",
        "M5",
        frame,
        long_probability=0.78,
        short_probability=0.22,
        chronos_direction=0.001,
    )
    assert signal.side == "long"

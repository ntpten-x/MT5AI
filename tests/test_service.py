from __future__ import annotations

import pandas as pd

from config import Settings
from modules.service import TradingBotService


class _DummyModel:
    def __init__(self, returned_threshold: float):
        self.returned_threshold = returned_threshold

    def effective_threshold(self, default: float) -> float:
        return float(self.returned_threshold)


def test_service_prefers_symbol_threshold_over_artifact_threshold():
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        model={
            "symbol_long_thresholds": {"GOLD": 0.18},
            "symbol_short_thresholds": {"GOLD": 0.21},
        },
    )
    service = TradingBotService.__new__(TradingBotService)
    service.settings = settings
    service.xgb_models = {
        ("GOLD", "long"): _DummyModel(0.77),
        ("GOLD", "short"): _DummyModel(0.88),
    }
    service._xgb_model = lambda symbol, target_side: service.xgb_models[(symbol, target_side)]

    long_threshold, short_threshold = service._xgb_thresholds("GOLD")

    assert long_threshold == 0.18
    assert short_threshold == 0.21


def test_service_uses_artifact_threshold_when_no_symbol_override():
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        model={"xgb_probability_threshold": 0.18},
    )
    service = TradingBotService.__new__(TradingBotService)
    service.settings = settings
    service.xgb_models = {
        ("GOLD", "long"): _DummyModel(0.77),
        ("GOLD", "short"): _DummyModel(0.88),
    }
    service._xgb_model = lambda symbol, target_side: service.xgb_models[(symbol, target_side)]

    long_threshold, short_threshold = service._xgb_thresholds("GOLD")

    assert long_threshold == 0.77
    assert short_threshold == 0.88


def test_service_chronos_gate_only_activates_in_selected_regime():
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        model={
            "chronos_enabled": True,
            "chronos_selective_enabled": True,
            "chronos_allowed_sessions": ["overlap"],
            "chronos_min_atr_pct": 0.0005,
            "chronos_max_atr_pct": 0.005,
            "chronos_max_abs_ema_trend_strength": 1.0,
            "chronos_max_spread_pct": 0.05,
        },
    )
    service = TradingBotService.__new__(TradingBotService)
    service.settings = settings

    active = service._chronos_gate_active(
        "GOLD",
        pd.DataFrame(
            [
                {
                    "atr_pct": 0.0012,
                    "ema_trend_strength": 0.4,
                    "spread_pct": 0.01,
                    "session_tokyo": 0,
                    "session_london": 1,
                    "session_new_york": 1,
                    "session_overlap": 1,
                }
            ]
        ),
    )
    blocked = service._chronos_gate_active(
        "GOLD",
        pd.DataFrame(
            [
                {
                    "atr_pct": 0.0001,
                    "ema_trend_strength": 1.4,
                    "spread_pct": 0.08,
                    "session_tokyo": 1,
                    "session_london": 0,
                    "session_new_york": 0,
                    "session_overlap": 0,
                }
            ]
        ),
    )

    assert active is True
    assert blocked is False


def test_service_reset_kill_switch_returns_previous_state(tmp_path):
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        bot={"kill_switch_path": str(tmp_path / "kill_switch_state.json")},
    )
    service = TradingBotService.__new__(TradingBotService)
    service.settings = settings
    service.database = type("Database", (), {"record_event": lambda *args, **kwargs: None})()
    service.notifier = type("Notifier", (), {"send_warning": lambda *args, **kwargs: True})()
    from modules.risk_control import RiskManager

    service.risk_manager = RiskManager(
        settings,
        type(
            "RiskDb",
            (),
            {
                "record_equity_snapshot": lambda *args, **kwargs: None,
                "get_daily_equity_baseline": lambda *args, **kwargs: None,
                "get_daily_high_watermark_equity": lambda *args, **kwargs: None,
                "count_consecutive_losses": lambda *args, **kwargs: 0,
            },
        )(),
    )
    service.risk_manager._trip_kill_switch(
        event_code="kill_switch_daily_drawdown",
        reason="halted",
        metric_name="daily_drawdown",
        metric_value=0.03,
        threshold=0.02,
    )

    result = service.reset_kill_switch()

    assert result["status"] == "reset"
    assert result["kill_switch_was_active"] is True
    assert result["previous"]["reason"] == "halted"
    assert service.risk_manager.kill_switch_active() is False

from __future__ import annotations

from config import Settings
from modules.backtest import BacktestEngine
from modules.signals import SignalEngine


class _DummyModel:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def effective_threshold(self, default: float) -> float:
        return float(self.threshold)


def test_backtest_engine_prefers_explicit_symbol_threshold_over_artifact_threshold():
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        model={
            "xgb_probability_threshold": 0.18,
            "symbol_long_thresholds": {"GOLD": 0.18},
            "symbol_short_thresholds": {"GOLD": 0.2},
        },
    )
    engine = BacktestEngine(settings, SignalEngine(settings))

    assert engine._resolved_threshold("GOLD", "long", _DummyModel(0.77)) == 0.18
    assert engine._resolved_threshold("GOLD", "short", _DummyModel(0.88)) == 0.2


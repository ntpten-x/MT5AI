from __future__ import annotations

from config import Settings
from modules.signals import SignalEngine
from modules.walk_forward import WalkForwardEngine


def _engine(tmp_path) -> WalkForwardEngine:
    settings = Settings(
        _env_file=None,
        bot={"data_dir": str(tmp_path / "data")},
        model={
            "walk_forward_min_positive_return_folds": 2,
            "walk_forward_min_positive_return_pct": 0.0,
        },
    )
    return WalkForwardEngine(settings, SignalEngine(settings))


def test_recommend_profile_promotes_consensus_from_positive_return_folds(tmp_path):
    engine = _engine(tmp_path)
    report = {
        "summary": {"mean_accuracy": 0.82, "mean_roc_auc": 0.68},
        "folds": [
            {
                "fold": 1,
                "threshold_tuning": {
                    "long_threshold": 0.53,
                    "short_threshold": 0.69,
                    "risk_per_trade": 0.0045,
                    "atr_stop_loss_mult": 3.3,
                    "atr_take_profit_mult": 4.6,
                    "sequence_probability_weight": 0.45,
                    "objective": 1.9,
                },
                "backtest": {"Total Return [%]": -0.05},
            },
            {
                "fold": 2,
                "threshold_tuning": {
                    "long_threshold": 0.55,
                    "short_threshold": 0.67,
                    "risk_per_trade": 0.005,
                    "atr_stop_loss_mult": 3.0,
                    "atr_take_profit_mult": 3.2,
                    "sequence_probability_weight": 0.45,
                    "objective": 3.74,
                },
                "backtest": {"Total Return [%]": 0.61},
            },
            {
                "fold": 5,
                "threshold_tuning": {
                    "long_threshold": 0.55,
                    "short_threshold": 0.62,
                    "risk_per_trade": 0.006,
                    "atr_stop_loss_mult": 3.1,
                    "atr_take_profit_mult": 4.3,
                    "sequence_probability_weight": 0.2,
                    "objective": -5.04,
                },
                "backtest": {"Total Return [%]": 1.04},
            },
        ],
    }

    profile = engine.recommend_profile("BTCUSD", report)

    assert profile["profile_accepted"] is True
    assert profile["profile_source"] == "walk_forward_consensus"
    assert profile["walk_forward_positive_return_folds"] == 2
    assert profile["walk_forward_selected_folds"] == [2, 5]
    assert profile["long_threshold"] == 0.55
    assert profile["short_threshold"] == 0.645
    assert profile["risk_per_trade"] == 0.0055
    assert profile["atr_stop_loss_mult"] == 3.05
    assert profile["atr_take_profit_mult"] == 3.75
    assert profile["sequence_probability_weight"] == 0.325


def test_recommend_profile_rejects_when_positive_return_folds_are_insufficient(tmp_path):
    engine = _engine(tmp_path)
    report = {
        "summary": {"mean_accuracy": 0.93, "mean_roc_auc": 0.68},
        "folds": [
            {
                "fold": 1,
                "threshold_tuning": {
                    "long_threshold": 0.61,
                    "short_threshold": 0.79,
                    "risk_per_trade": 0.005,
                    "atr_stop_loss_mult": 2.6,
                    "atr_take_profit_mult": 2.0,
                    "sequence_probability_weight": 0.1,
                    "objective": -15.0,
                },
                "backtest": {"Total Return [%]": 0.0},
            },
            {
                "fold": 2,
                "threshold_tuning": {
                    "long_threshold": 0.61,
                    "short_threshold": 0.79,
                    "risk_per_trade": 0.005,
                    "atr_stop_loss_mult": 2.6,
                    "atr_take_profit_mult": 2.0,
                    "sequence_probability_weight": 0.1,
                    "objective": -15.0,
                },
                "backtest": {"Total Return [%]": 0.0},
            },
        ],
    }

    profile = engine.recommend_profile("GOLD", report)

    assert profile["profile_accepted"] is False
    assert profile["profile_source"] == "walk_forward_rejected"
    assert profile["walk_forward_positive_return_folds"] == 0
    assert "rejection_reason" in profile
    assert "long_threshold" not in profile


def test_recommend_profile_rejects_when_regime_coverage_is_missing(tmp_path):
    engine = _engine(tmp_path)
    report = {
        "summary": {
            "mean_accuracy": 0.81,
            "mean_roc_auc": 0.67,
            "regime_counts": {"sideway": 2, "mixed": 1},
        },
        "folds": [
            {
                "fold": 1,
                "threshold_tuning": {
                    "long_threshold": 0.55,
                    "short_threshold": 0.67,
                    "risk_per_trade": 0.005,
                    "atr_stop_loss_mult": 3.0,
                    "atr_take_profit_mult": 3.2,
                    "sequence_probability_weight": 0.45,
                    "objective": 3.74,
                },
                "backtest": {"Total Return [%]": 0.61},
            },
            {
                "fold": 2,
                "threshold_tuning": {
                    "long_threshold": 0.56,
                    "short_threshold": 0.66,
                    "risk_per_trade": 0.005,
                    "atr_stop_loss_mult": 3.1,
                    "atr_take_profit_mult": 3.4,
                    "sequence_probability_weight": 0.35,
                    "objective": 2.15,
                },
                "backtest": {"Total Return [%]": 0.48},
            },
        ],
    }

    profile = engine.recommend_profile("GOLD", report)

    assert profile["profile_accepted"] is False
    assert profile["walk_forward_regime_coverage_ok"] is False
    assert profile["walk_forward_missing_regimes"] == ["strong_trend"]
    assert "Missing required market regimes" in profile["rejection_reason"]

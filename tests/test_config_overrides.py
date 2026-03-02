from __future__ import annotations

import json

from config import Settings


def test_settings_symbol_overrides_resolve_threshold_margin_and_risk(tmp_path):
    settings = Settings(
        _env_file=None,
        model={
            "xgb_probability_threshold": 0.58,
            "probability_dominance_margin": 0.30,
            "symbol_long_thresholds": {"BTCUSD": 0.52},
            "symbol_short_thresholds": {"BTCUSD": 0.54},
            "symbol_probability_dominance_margins": {"BTCUSD": 0.20},
            "optimization_profile": str(tmp_path / "profile.json"),
        },
        risk={
            "risk_per_trade": 0.005,
            "symbol_risk_per_trade": {"BTCUSD": 0.002},
        },
    )

    assert settings.threshold_for("BTCUSD", "long") == 0.52
    assert settings.threshold_for("BTCUSD", "short") == 0.54
    assert settings.threshold_for("ETHUSD", "long") == 0.58
    assert settings.dominance_margin_for("BTCUSD") == 0.20
    assert settings.dominance_margin_for("ETHUSD") == 0.30
    assert settings.risk_per_trade_for("BTCUSD") == 0.002
    assert settings.risk_per_trade_for("ETHUSD") == 0.005


def test_settings_load_optimization_profile_overrides(tmp_path):
    profile_path = tmp_path / "BTCUSD_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "long_threshold": 0.57,
                "short_threshold": 0.56,
                "risk_per_trade": 0.003,
                "atr_stop_loss_mult": 2.4,
                "atr_take_profit_mult": 4.1,
                "sequence_probability_weight": 0.42,
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["BTCUSD"]},
        model={
            "xgb_probability_threshold": 0.60,
            "optimization_profile": str(profile_path),
        },
        risk={
            "risk_per_trade": 0.005,
            "atr_stop_loss_mult": 1.8,
            "atr_take_profit_mult": 3.0,
        },
    )

    assert settings.threshold_for("BTCUSD", "long") == 0.57
    assert settings.threshold_for("BTCUSD", "short") == 0.56
    assert settings.risk_per_trade_for("BTCUSD") == 0.003
    assert settings.atr_stop_loss_mult_for("BTCUSD") == 2.4
    assert settings.atr_take_profit_mult_for("BTCUSD") == 4.1
    assert settings.sequence_probability_weight_for("BTCUSD") == 0.42


def test_settings_explicit_symbol_probability_weights_override_profile(tmp_path):
    profile_path = tmp_path / "GOLD_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "sequence_probability_weight": 0.42,
                "transformer_probability_weight": 0.25,
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        model={
            "optimization_profile": str(profile_path),
            "sequence_probability_weight": 0.30,
            "transformer_probability_weight": 0.15,
            "symbol_sequence_probability_weights": {"GOLD": 0.35},
            "symbol_transformer_probability_weights": {"GOLD": 0.10},
        },
    )

    assert settings.sequence_probability_weight_for("GOLD") == 0.35
    assert settings.transformer_probability_weight_for("GOLD") == 0.10


def test_settings_explicit_lightgbm_weight_override_profile(tmp_path):
    profile_path = tmp_path / "GOLD_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "lightgbm_probability_weight": 0.27,
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        model={
            "optimization_profile": str(profile_path),
            "lightgbm_probability_weight": 0.12,
            "symbol_lightgbm_probability_weights": {"GOLD": 0.18},
        },
    )

    assert settings.lightgbm_probability_weight_for("GOLD") == 0.18
    assert settings.model.lightgbm_probability_weight == 0.12


def test_settings_symbol_execution_overrides_resolve_risk_runtime_values(tmp_path):
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        risk={
            "auto_breakeven_rr_default": 1.0,
            "time_exit_bars": 5,
            "time_exit_min_progress_rr": 0.3,
            "trailing_trigger_rr": 0.0,
            "trailing_lock_rr": 0.0,
            "symbol_auto_breakeven_rr": {"GOLD": 0.75},
            "symbol_time_exit_bars": {"GOLD": 8},
            "symbol_time_exit_min_progress_rr": {"GOLD": 0.2},
            "symbol_trailing_trigger_rr": {"GOLD": 1.2},
            "symbol_trailing_lock_rr": {"GOLD": 0.35},
        },
    )

    assert settings.auto_breakeven_rr_for("GOLD") == 0.75
    assert settings.time_exit_bars_for("GOLD") == 8
    assert settings.time_exit_min_progress_rr_for("GOLD") == 0.2
    assert settings.trailing_trigger_rr_for("GOLD") == 1.2
    assert settings.trailing_lock_rr_for("GOLD") == 0.35


def test_settings_ignore_rejected_optimization_profile(tmp_path):
    profile_path = tmp_path / "GOLD_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_accepted": False,
                "long_threshold": 0.74,
                "short_threshold": 0.75,
                "risk_per_trade": 0.001,
                "atr_stop_loss_mult": 3.2,
                "atr_take_profit_mult": 4.4,
                "sequence_probability_weight": 0.55,
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["GOLD"]},
        model={
            "xgb_probability_threshold": 0.60,
            "optimization_profile": str(profile_path),
        },
        risk={
            "risk_per_trade": 0.005,
            "atr_stop_loss_mult": 1.8,
            "atr_take_profit_mult": 3.0,
        },
    )

    assert settings.threshold_for("GOLD", "long") == 0.60
    assert settings.threshold_for("GOLD", "short") == 0.60
    assert settings.risk_per_trade_for("GOLD") == 0.005
    assert settings.atr_stop_loss_mult_for("GOLD") == 1.8
    assert settings.atr_take_profit_mult_for("GOLD") == 3.0
    assert settings.sequence_probability_weight_for("GOLD") == settings.model.sequence_probability_weight


def test_settings_parse_confirmation_timeframes_and_deduplicate():
    settings = Settings(
        _env_file=None,
        trading={
            "timeframe": "m5",
            "confirmation_timeframes": "m15,h1,M15",
        },
    )

    assert settings.trading.timeframe == "M5"
    assert settings.trading.confirmation_timeframes == ["M15", "H1"]
    assert settings.trading_timeframes() == ["M5", "M15", "H1"]

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import Settings
from modules.spread_profile import (
    SpreadProfileManager,
    classify_market_regime,
    classify_trading_session,
)


def _sample_spread_frame(rows: int = 2_000) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    spreads = []
    for timestamp in times:
        hour = timestamp.hour
        if hour in {23, 0, 1}:
            spreads.append(48 + (hour == 23) * 5)
        elif 13 <= hour < 16:
            spreads.append(38)
        elif 7 <= hour < 13:
            spreads.append(40)
        elif 16 <= hour < 22:
            spreads.append(37)
        else:
            spreads.append(43)
    return pd.DataFrame({"time": times, "spread": np.array(spreads, dtype=float)})


def _constant_spread_frame(spread: float, rows: int = 2_000) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    return pd.DataFrame({"time": times, "spread": np.full(rows, spread, dtype=float)})


def test_classify_trading_session_handles_rollover_and_overlap():
    rollover_hours = {23, 0, 1}
    assert classify_trading_session(23, rollover_hours) == "rollover"
    assert classify_trading_session(14, rollover_hours) == "overlap"
    assert classify_trading_session(9, rollover_hours) == "london"
    assert classify_trading_session(18, rollover_hours) == "new_york"
    assert classify_trading_session(4, rollover_hours) == "tokyo"


def test_classify_market_regime_handles_weekend_open_rollover_and_weekday():
    rollover_hours = {23, 0, 1}
    weekend_days = {6, 0}
    weekend_hours = {22, 23, 0, 1}

    assert (
        classify_market_regime(
            datetime(2026, 1, 4, 23, 0, tzinfo=timezone.utc),
            rollover_hours,
            weekend_days,
            weekend_hours,
        )
        == "weekend_open"
    )
    assert (
        classify_market_regime(
            datetime(2026, 1, 7, 23, 0, tzinfo=timezone.utc),
            rollover_hours,
            weekend_days,
            weekend_hours,
        )
        == "rollover"
    )
    assert (
        classify_market_regime(
            datetime(2026, 1, 7, 14, 0, tzinfo=timezone.utc),
            rollover_hours,
            weekend_days,
            weekend_hours,
        )
        == "weekday"
    )


def test_spread_profile_manager_builds_dynamic_limits(tmp_path):
    settings = Settings(
        risk={
            "spread_limit_points": 150,
            "news_spread_limit_points": 100,
            "extreme_spread_limit_points": 220,
        },
        spread_profile={
            "enabled": True,
            "symbols": ["GOLD"],
            "cache_path": str(tmp_path / "spread_profiles.json"),
            "min_rows": 500,
            "rollover_hours_utc": [23, 0, 1],
        },
    )
    manager = SpreadProfileManager(settings)
    frame = _sample_spread_frame()

    profile = manager.build_profile("GOLD", "M5", frame)
    decision = manager.evaluate(
        "GOLD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
    )

    assert profile is not None
    assert "weekday" in profile.regime_buckets
    assert "weekend_open" in profile.regime_buckets
    assert "overlap" in profile.session_buckets
    assert decision.profile_ready is True
    assert decision.source == "regime_session_hour_profile"
    assert decision.base_regime_name == "weekday"
    assert decision.runtime_regime_name == "weekday"
    assert decision.soft_limit_points < settings.risk.spread_limit_points
    assert decision.news_limit_points <= decision.soft_limit_points
    assert decision.extreme_limit_points > decision.soft_limit_points


def test_spread_profile_manager_tightens_limits_in_news_window(tmp_path):
    settings = Settings(
        risk={
            "spread_limit_points": 150,
            "news_spread_limit_points": 100,
            "extreme_spread_limit_points": 220,
        },
        spread_profile={
            "enabled": True,
            "symbols": ["GOLD"],
            "cache_path": str(tmp_path / "spread_profiles.json"),
            "min_rows": 500,
            "rollover_hours_utc": [23, 0, 1],
            "weekend_open_days_utc": [6, 0],
            "weekend_open_hours_utc": [22, 23, 0, 1],
        },
    )
    manager = SpreadProfileManager(settings)
    frame = _sample_spread_frame()

    normal = manager.evaluate(
        "GOLD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
        news_phase=None,
    )
    pre_news = manager.evaluate(
        "GOLD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
        news_phase="pre_news",
    )
    pre_news_close_only = manager.evaluate(
        "GOLD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
        news_phase="pre_news_close_only",
    )
    release = manager.evaluate(
        "GOLD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
        news_phase="release_minute",
    )
    freeze = manager.evaluate(
        "GOLD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
        news_phase="post_release_freeze",
    )
    post = manager.evaluate(
        "GOLD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
        news_phase="post_news_cooldown",
    )
    reentry = manager.evaluate(
        "GOLD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
        news_phase="post_news_reentry",
    )

    assert pre_news.runtime_regime_name == "pre_news"
    assert pre_news.source == "pre_news_profile"
    assert pre_news_close_only.runtime_regime_name == "pre_news_close_only"
    assert pre_news_close_only.source == "pre_news_close_only_profile"
    assert release.runtime_regime_name == "release_minute"
    assert release.source == "release_minute_profile"
    assert freeze.runtime_regime_name == "post_release_freeze"
    assert freeze.source == "post_release_freeze_profile"
    assert post.runtime_regime_name == "post_news_cooldown"
    assert post.source == "post_news_cooldown_profile"
    assert reentry.runtime_regime_name == "post_news_reentry"
    assert reentry.source == "post_news_reentry_profile"
    assert pre_news.soft_limit_points <= normal.soft_limit_points
    assert pre_news_close_only.soft_limit_points <= pre_news.soft_limit_points
    assert release.soft_limit_points <= pre_news_close_only.soft_limit_points
    assert freeze.soft_limit_points <= pre_news.soft_limit_points
    assert post.soft_limit_points <= normal.soft_limit_points
    assert reentry.soft_limit_points <= normal.soft_limit_points


def test_spread_profile_manager_falls_back_when_symbol_disabled(tmp_path):
    settings = Settings(
        risk={
            "spread_limit_points": 150,
            "news_spread_limit_points": 100,
            "extreme_spread_limit_points": 220,
        },
        spread_profile={
            "enabled": True,
            "symbols": ["GOLD"],
            "cache_path": str(tmp_path / "spread_profiles.json"),
        },
    )
    manager = SpreadProfileManager(settings)

    decision = manager.evaluate(
        "EURUSD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
    )

    assert decision.profile_ready is False
    assert decision.source == "global_defaults"
    assert decision.soft_limit_points == 150


def test_spread_profile_manager_uses_crypto_caps_for_high_spread_symbols(tmp_path):
    settings = Settings(
        trading={
            "symbols": ["BTCUSD"],
            "crypto_symbols": ["BTCUSD"],
        },
        risk={
            "spread_limit_points": 150,
            "news_spread_limit_points": 100,
            "extreme_spread_limit_points": 220,
            "crypto_spread_limit_points": 6000,
            "crypto_news_spread_limit_points": 6000,
            "crypto_extreme_spread_limit_points": 9000,
        },
        spread_profile={
            "enabled": True,
            "symbols": ["BTCUSD"],
            "cache_path": str(tmp_path / "spread_profiles.json"),
            "min_rows": 500,
        },
    )
    manager = SpreadProfileManager(settings)
    frame = _constant_spread_frame(5000.0)

    decision = manager.evaluate(
        "BTCUSD",
        "M5",
        current_time=datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc),
        frame=frame,
    )

    assert decision.profile_ready is True
    assert decision.soft_limit_points >= 5000
    assert decision.news_limit_points >= 5000
    assert decision.extreme_limit_points >= 5015

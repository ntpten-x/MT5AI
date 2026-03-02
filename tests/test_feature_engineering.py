from __future__ import annotations

import numpy as np
import pandas as pd

from modules.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_frame,
    build_supervised_frame,
    build_training_frame,
    select_feature_columns,
)


def _sample_market_frame(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 1.10 + np.cumsum(rng.normal(0, 0.001, size=rows))
    open_ = close + rng.normal(0, 0.0004, size=rows)
    high = np.maximum(open_, close) + rng.uniform(0.0002, 0.001, size=rows)
    low = np.minimum(open_, close) - rng.uniform(0.0002, 0.001, size=rows)
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(100, 1000, size=rows),
            "spread": rng.integers(5, 20, size=rows),
            "real_volume": rng.integers(0, 10, size=rows),
        }
    )


def _sample_context_frame(rows: int = 80, freq: str = "15min", drift: float = 0.0015) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 2500.0 + np.cumsum(rng.normal(drift, 0.8, size=rows))
    open_ = close + rng.normal(0, 0.3, size=rows)
    high = np.maximum(open_, close) + rng.uniform(0.2, 0.9, size=rows)
    low = np.minimum(open_, close) - rng.uniform(0.2, 0.9, size=rows)
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=rows, freq=freq, tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(100, 1000, size=rows),
            "spread": rng.integers(20, 60, size=rows),
            "real_volume": rng.integers(0, 10, size=rows),
        }
    )


def test_build_feature_frame_contains_expected_columns():
    frame = build_feature_frame(_sample_market_frame())
    for column in FEATURE_COLUMNS:
        assert column in frame.columns


def test_build_feature_frame_adds_multi_timeframe_context_features():
    frame = _sample_market_frame()
    context_frames = {
        "M15": _sample_context_frame(rows=90, freq="15min", drift=0.8),
        "H1": _sample_context_frame(rows=40, freq="1h", drift=1.2),
    }

    features = build_feature_frame(frame, symbol="GOLD", timeframe_context_frames=context_frames)

    assert "mtf_alignment_score" in features.columns
    assert "mtf_rsi_mean" in features.columns
    assert features["mtf_context_count"].iloc[-1] >= 1.0
    assert abs(float(features["mtf_alignment_score"].iloc[-1])) <= 1.0


def test_build_training_frame_generates_target_column():
    frame = build_training_frame(_sample_market_frame(), horizon=3, edge_bps=4)
    assert "target" in frame.columns
    assert "future_return" in frame.columns
    assert not frame.empty


def test_build_supervised_frame_generates_dual_targets():
    frame = build_supervised_frame(_sample_market_frame(), horizon=3, edge_bps=4)
    assert "target_long" in frame.columns
    assert "target_short" in frame.columns
    assert not frame.empty


def test_select_feature_columns_returns_forced_and_respects_limits():
    frame = build_training_frame(_sample_market_frame(rows=320), horizon=3, edge_bps=4)

    selected = select_feature_columns(
        frame,
        target_column="target",
        min_features=8,
        max_features=12,
        forced_features=["ema_fast", "atr_14", "not_a_feature"],
        correlation_threshold=0.90,
        min_mutual_info=0.0,
    )

    assert "ema_fast" in selected
    assert "atr_14" in selected
    assert "not_a_feature" not in selected
    assert 8 <= len(selected) <= 12
    assert len(selected) == len(set(selected))

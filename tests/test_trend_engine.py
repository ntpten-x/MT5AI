from __future__ import annotations

import pandas as pd

from invest_advisor_bot.analysis.trend_engine import evaluate_trend


def test_evaluate_trend_detects_uptrend() -> None:
    close = pd.Series([100, 101, 102, 104, 106, 108, 110, 113, 116, 120], dtype=float)
    frame = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 1,
            "low": close - 2,
            "close": close,
        }
    )

    assessment = evaluate_trend(frame, ticker="SPY", ema_fast_span=3, ema_slow_span=5, rsi_period=3)

    assert assessment.ticker == "SPY"
    assert assessment.direction == "uptrend"
    assert assessment.score > 0

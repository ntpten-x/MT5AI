from __future__ import annotations

import pandas as pd

from invest_advisor_bot.analysis.technical_indicators import (
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_support_resistance,
)


def test_indicator_functions_return_expected_shapes() -> None:
    close = pd.Series([100, 101, 102, 103, 104, 103, 105, 106, 108, 110], dtype=float)
    frame = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )

    ema = calculate_ema(close, span=3)
    rsi = calculate_rsi(close, period=3)
    macd = calculate_macd(close, fast_span=3, slow_span=6, signal_span=2)
    levels = calculate_support_resistance(frame, lookback=5, pivot_window=1)

    assert len(ema) == len(close)
    assert len(rsi) == len(close)
    assert list(macd.columns) == ["macd", "macd_signal", "macd_hist"]
    assert levels.current_price == float(close.iloc[-1])

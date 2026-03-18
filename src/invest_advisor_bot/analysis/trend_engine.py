from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .technical_indicators import (
    SupportResistanceLevels,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_support_resistance,
)

TrendDirection = Literal["uptrend", "downtrend", "sideways"]


@dataclass(slots=True, frozen=True)
class TrendAssessment:
    ticker: str | None
    direction: TrendDirection
    score: float
    current_price: float
    ema_fast: float
    ema_slow: float
    ema_gap_pct: float
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    support_resistance: SupportResistanceLevels
    reasons: list[str]


def evaluate_trend(
    frame: pd.DataFrame,
    *,
    ticker: str | None = None,
    ema_fast_span: int = 12,
    ema_slow_span: int = 26,
    rsi_period: int = 14,
    support_resistance_lookback: int = 20,
    pivot_window: int = 3,
) -> TrendAssessment:
    """Assess whether an asset is trending up, down, or sideways."""

    if "close" not in frame.columns:
        raise KeyError("frame is missing required column: close")
    if frame.empty:
        raise ValueError("frame must not be empty")

    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError("frame does not contain usable close prices")

    ema_fast = calculate_ema(close, span=ema_fast_span)
    ema_slow = calculate_ema(close, span=ema_slow_span)
    rsi = calculate_rsi(close, period=rsi_period)
    macd_frame = calculate_macd(
        close,
        fast_span=ema_fast_span,
        slow_span=ema_slow_span,
        signal_span=9,
    )
    levels = calculate_support_resistance(
        frame.loc[close.index],
        lookback=support_resistance_lookback,
        pivot_window=pivot_window,
    )

    latest_price = float(close.iloc[-1])
    latest_ema_fast = float(ema_fast.iloc[-1])
    latest_ema_slow = float(ema_slow.iloc[-1])
    ema_gap_pct = (
        (latest_ema_fast - latest_ema_slow) / latest_price if latest_price else 0.0
    )
    latest_rsi = _optional_scalar(rsi.iloc[-1] if not rsi.empty else None)
    latest_macd = _optional_scalar(macd_frame["macd"].iloc[-1] if not macd_frame.empty else None)
    latest_macd_signal = _optional_scalar(
        macd_frame["macd_signal"].iloc[-1] if not macd_frame.empty else None
    )
    latest_macd_hist = _optional_scalar(macd_frame["macd_hist"].iloc[-1] if not macd_frame.empty else None)

    score = 0.0
    reasons: list[str] = []

    if latest_price > latest_ema_fast > latest_ema_slow:
        score += 2.0
        reasons.append("price_above_fast_and_slow_ema")
    elif latest_price < latest_ema_fast < latest_ema_slow:
        score -= 2.0
        reasons.append("price_below_fast_and_slow_ema")
    else:
        reasons.append("price_mixed_vs_ema")

    if ema_gap_pct > 0.003:
        score += 1.0
        reasons.append("ema_spread_bullish")
    elif ema_gap_pct < -0.003:
        score -= 1.0
        reasons.append("ema_spread_bearish")
    else:
        reasons.append("ema_spread_flat")

    if latest_rsi is not None:
        if latest_rsi >= 55.0:
            score += 1.0
            reasons.append("rsi_above_55")
        elif latest_rsi <= 45.0:
            score -= 1.0
            reasons.append("rsi_below_45")
        else:
            reasons.append("rsi_neutral")

    if latest_macd is not None and latest_macd_signal is not None:
        if latest_macd > latest_macd_signal and (latest_macd_hist or 0.0) > 0.0:
            score += 1.0
            reasons.append("macd_bullish")
        elif latest_macd < latest_macd_signal and (latest_macd_hist or 0.0) < 0.0:
            score -= 1.0
            reasons.append("macd_bearish")
        else:
            reasons.append("macd_neutral")

    direction: TrendDirection
    if score >= 2.5:
        direction = "uptrend"
    elif score <= -2.5:
        direction = "downtrend"
    else:
        direction = "sideways"

    return TrendAssessment(
        ticker=ticker,
        direction=direction,
        score=score,
        current_price=latest_price,
        ema_fast=latest_ema_fast,
        ema_slow=latest_ema_slow,
        ema_gap_pct=ema_gap_pct,
        rsi=latest_rsi,
        macd=latest_macd,
        macd_signal=latest_macd_signal,
        macd_hist=latest_macd_hist,
        support_resistance=levels,
        reasons=reasons,
    )


def _optional_scalar(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


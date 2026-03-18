from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PriceColumn = Literal["open", "high", "low", "close"]


@dataclass(slots=True, frozen=True)
class SupportResistanceLevels:
    current_price: float
    nearest_support: float | None
    nearest_resistance: float | None
    supports: list[float]
    resistances: list[float]
    rolling_support: float | None
    rolling_resistance: float | None


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Return exponential moving average for a price series."""

    normalized = _to_numeric_series(series, name=series.name or "price")
    if span <= 0:
        raise ValueError("span must be greater than 0")
    return normalized.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return Wilder RSI for a close-price series."""

    normalized = _to_numeric_series(series, name=series.name or "close")
    if period <= 0:
        raise ValueError("period must be greater than 0")

    delta = normalized.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi.clip(lower=0.0, upper=100.0)


def calculate_macd(
    series: pd.Series,
    *,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Return MACD line, signal line, and histogram."""

    normalized = _to_numeric_series(series, name=series.name or "close")
    if fast_span <= 0 or slow_span <= 0 or signal_span <= 0:
        raise ValueError("MACD spans must be greater than 0")
    if fast_span >= slow_span:
        raise ValueError("fast_span must be smaller than slow_span")

    ema_fast = calculate_ema(normalized, span=fast_span)
    ema_slow = calculate_ema(normalized, span=slow_span)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        },
        index=normalized.index,
    )


def calculate_support_resistance(
    frame: pd.DataFrame,
    *,
    lookback: int = 20,
    pivot_window: int = 3,
    max_levels: int = 3,
    price_column: PriceColumn = "close",
) -> SupportResistanceLevels:
    """Estimate nearby support and resistance from rolling range and swing pivots."""

    normalized = _validate_ohlc_frame(frame)
    if lookback <= 0:
        raise ValueError("lookback must be greater than 0")
    if pivot_window <= 0:
        raise ValueError("pivot_window must be greater than 0")
    if max_levels <= 0:
        raise ValueError("max_levels must be greater than 0")

    current_price = float(normalized[price_column].iloc[-1])
    rolling_support = _as_optional_float(normalized["low"].rolling(lookback, min_periods=1).min().iloc[-1])
    rolling_resistance = _as_optional_float(normalized["high"].rolling(lookback, min_periods=1).max().iloc[-1])

    window_size = (pivot_window * 2) + 1
    pivot_high_mask = normalized["high"] == normalized["high"].rolling(
        window=window_size,
        center=True,
        min_periods=window_size,
    ).max()
    pivot_low_mask = normalized["low"] == normalized["low"].rolling(
        window=window_size,
        center=True,
        min_periods=window_size,
    ).min()

    pivot_highs = sorted({_round_level(value) for value in normalized.loc[pivot_high_mask, "high"].dropna().tolist()})
    pivot_lows = sorted({_round_level(value) for value in normalized.loc[pivot_low_mask, "low"].dropna().tolist()})

    supports = [level for level in pivot_lows if level <= current_price]
    resistances = [level for level in pivot_highs if level >= current_price]

    support_levels = supports[-max_levels:]
    resistance_levels = resistances[:max_levels]

    nearest_support = support_levels[-1] if support_levels else rolling_support
    nearest_resistance = resistance_levels[0] if resistance_levels else rolling_resistance

    if nearest_support is None and rolling_support is not None:
        support_levels = [_round_level(rolling_support)]
    if nearest_resistance is None and rolling_resistance is not None:
        resistance_levels = [_round_level(rolling_resistance)]

    return SupportResistanceLevels(
        current_price=current_price,
        nearest_support=_as_optional_float(nearest_support),
        nearest_resistance=_as_optional_float(nearest_resistance),
        supports=[float(level) for level in support_levels],
        resistances=[float(level) for level in resistance_levels],
        rolling_support=rolling_support,
        rolling_resistance=rolling_resistance,
    )


def _validate_ohlc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"high", "low", "close"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("frame must not be empty")

    normalized = frame.copy()
    for column in required_columns.union({"open"}):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["high", "low", "close"])
    if normalized.empty:
        raise ValueError("frame does not contain usable OHLC data")
    return normalized


def _to_numeric_series(series: pd.Series, *, name: str) -> pd.Series:
    if series.empty:
        raise ValueError(f"{name} series must not be empty")
    normalized = pd.to_numeric(series.copy(), errors="coerce")
    normalized = normalized.dropna()
    if normalized.empty:
        raise ValueError(f"{name} series does not contain usable numeric values")
    return normalized


def _round_level(value: float) -> float:
    return round(float(value), 4)


def _as_optional_float(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


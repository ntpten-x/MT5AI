from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional fallback
    try:
        import pandas_ta_classic as ta
    except ImportError:  # pragma: no cover - optional fallback
        ta = None

BASE_FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "return_20",
    "ema_fast",
    "ema_slow",
    "ema_gap_pct",
    "ema_trend_strength",
    "rsi_14",
    "rsi_7",
    "rsi_delta_3",
    "atr_14",
    "atr_pct",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_mid",
    "bb_upper_gap",
    "bb_lower_gap",
    "bb_width",
    "funding_rate",
    "btc_eth_corr_48",
    "btc_dxy_corr_48",
    "volatility_20",
    "close_zscore_20",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range_to_atr",
    "roll_high_gap_20",
    "roll_low_gap_20",
    "spread_pct",
    "volume_change_1",
    "volume_zscore_20",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "session_tokyo",
    "session_london",
    "session_new_york",
    "session_overlap",
]

CANDIDATE_FEATURE_COLUMNS = [
    "roc_10",
    "stoch_k_14",
    "stoch_d_14",
    "cci_20",
    "willr_14",
    "mfi_14",
    "donchian_width_20",
    "obv_zscore_20",
    "volume_price_trend",
    "atr_accel_3",
]

FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *CANDIDATE_FEATURE_COLUMNS]


def _ensure_time_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "time" in frame.columns:
        return frame.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        copy = frame.reset_index().rename(columns={frame.index.name or "index": "time"})
        return copy
    raise KeyError("Input frame must contain a `time` column or a DatetimeIndex")


def _manual_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _manual_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _manual_stoch(frame: pd.DataFrame, period: int = 14, signal: int = 3) -> tuple[pd.Series, pd.Series]:
    rolling_low = frame["low"].rolling(period).min()
    rolling_high = frame["high"].rolling(period).max()
    denominator = (rolling_high - rolling_low).replace(0, np.nan)
    stoch_k = ((frame["close"] - rolling_low) / denominator) * 100.0
    stoch_d = stoch_k.rolling(signal).mean()
    return stoch_k, stoch_d


def _manual_cci(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda values: np.mean(np.abs(values - np.mean(values))), raw=True)
    return (typical_price - sma) / (0.015 * mad.replace(0, np.nan))


def _manual_willr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    rolling_low = frame["low"].rolling(period).min()
    rolling_high = frame["high"].rolling(period).max()
    denominator = (rolling_high - rolling_low).replace(0, np.nan)
    return -100.0 * ((rolling_high - frame["close"]) / denominator)


def _manual_mfi(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["tick_volume"] if "tick_volume" in frame.columns else pd.Series(0.0, index=frame.index)
    raw_flow = typical_price * volume
    direction = typical_price.diff()
    positive_flow = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
    negative_flow = raw_flow.where(direction < 0, 0.0).abs().rolling(period).sum()
    ratio = positive_flow / negative_flow.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def _aligned_series_by_time(base_time: pd.Series, reference: pd.DataFrame, column: str) -> pd.Series:
    if reference.empty or column not in reference.columns:
        return pd.Series(np.nan, index=base_time.index)
    if "time" not in reference.columns:
        return pd.Series(np.nan, index=base_time.index)
    indexed = reference[["time", column]].copy()
    indexed = indexed.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    aligned = indexed.set_index("time")[column]
    base_index = pd.to_datetime(base_time, utc=True)
    aligned = aligned.reindex(base_index, method="ffill")
    return pd.Series(aligned.to_numpy(), index=base_time.index)


def build_feature_frame(
    frame: pd.DataFrame,
    symbol: str | None = None,
    reference_frames: dict[str, pd.DataFrame] | None = None,
    funding_rate: float | None = None,
) -> pd.DataFrame:
    data = _ensure_time_column(frame)
    data = data.sort_values("time").reset_index(drop=True)
    symbol_upper = (symbol or "").upper()
    refs = {key.upper(): value for key, value in (reference_frames or {}).items()}

    if ta is not None:
        data["ema_fast"] = ta.ema(data["close"], length=12)
        data["ema_slow"] = ta.ema(data["close"], length=26)
        data["rsi_14"] = ta.rsi(data["close"], length=14)
        data["rsi_7"] = ta.rsi(data["close"], length=7)
        data["atr_14"] = ta.atr(data["high"], data["low"], data["close"], length=14)
        macd_frame = ta.macd(data["close"], fast=12, slow=26, signal=9)
        if macd_frame is not None and not macd_frame.empty:
            data["macd"] = macd_frame.iloc[:, 0]
            data["macd_hist"] = macd_frame.iloc[:, 1]
            data["macd_signal"] = macd_frame.iloc[:, 2]
        else:
            data["macd"] = data["close"].ewm(span=12, adjust=False).mean() - data["close"].ewm(
                span=26, adjust=False
            ).mean()
            data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
            data["macd_hist"] = data["macd"] - data["macd_signal"]
        bbands = ta.bbands(data["close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            data["bb_upper"] = bbands.iloc[:, 0]
            data["bb_mid"] = bbands.iloc[:, 1]
            data["bb_lower"] = bbands.iloc[:, 2]
        else:
            data["bb_mid"] = data["close"].rolling(20).mean()
            rolling_std_20 = data["close"].rolling(20).std()
            data["bb_upper"] = data["bb_mid"] + (rolling_std_20 * 2)
            data["bb_lower"] = data["bb_mid"] - (rolling_std_20 * 2)
        stoch_frame = ta.stoch(data["high"], data["low"], data["close"], k=14, d=3, smooth_k=3)
        if stoch_frame is not None and not stoch_frame.empty:
            data["stoch_k_14"] = stoch_frame.iloc[:, 0]
            data["stoch_d_14"] = stoch_frame.iloc[:, 1]
        else:
            data["stoch_k_14"], data["stoch_d_14"] = _manual_stoch(data, period=14, signal=3)
        cci_series = ta.cci(data["high"], data["low"], data["close"], length=20)
        data["cci_20"] = cci_series if cci_series is not None else _manual_cci(data, period=20)
        willr_series = ta.willr(data["high"], data["low"], data["close"], length=14)
        data["willr_14"] = willr_series if willr_series is not None else _manual_willr(data, period=14)
        mfi_series = (
            ta.mfi(data["high"], data["low"], data["close"], data["tick_volume"], length=14)
            if "tick_volume" in data.columns
            else None
        )
        data["mfi_14"] = mfi_series if mfi_series is not None else _manual_mfi(data, period=14)
    else:
        data["ema_fast"] = data["close"].ewm(span=12, adjust=False).mean()
        data["ema_slow"] = data["close"].ewm(span=26, adjust=False).mean()
        data["rsi_14"] = _manual_rsi(data["close"], period=14)
        data["rsi_7"] = _manual_rsi(data["close"], period=7)
        data["atr_14"] = _manual_atr(data, period=14)
        data["macd"] = data["ema_fast"] - data["ema_slow"]
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]
        data["bb_mid"] = data["close"].rolling(20).mean()
        rolling_std_20 = data["close"].rolling(20).std()
        data["bb_upper"] = data["bb_mid"] + (rolling_std_20 * 2)
        data["bb_lower"] = data["bb_mid"] - (rolling_std_20 * 2)
        data["stoch_k_14"], data["stoch_d_14"] = _manual_stoch(data, period=14, signal=3)
        data["cci_20"] = _manual_cci(data, period=20)
        data["willr_14"] = _manual_willr(data, period=14)
        data["mfi_14"] = _manual_mfi(data, period=14)

    data["return_1"] = data["close"].pct_change(1)
    data["return_3"] = data["close"].pct_change(3)
    data["return_5"] = data["close"].pct_change(5)
    data["return_10"] = data["close"].pct_change(10)
    data["return_20"] = data["close"].pct_change(20)
    data["ema_gap_pct"] = (data["ema_fast"] - data["ema_slow"]) / data["close"].replace(0, np.nan)
    data["ema_trend_strength"] = (data["ema_fast"] - data["ema_slow"]) / data["atr_14"].replace(0, np.nan)
    data["rsi_delta_3"] = data["rsi_14"].diff(3)
    data["bb_upper_gap"] = (data["bb_upper"] - data["close"]) / data["close"].replace(0, np.nan)
    data["bb_lower_gap"] = (data["close"] - data["bb_lower"]) / data["close"].replace(0, np.nan)
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["close"].replace(0, np.nan)
    data["funding_rate"] = float(funding_rate or 0.0)
    data["volatility_20"] = data["return_1"].rolling(20).std()
    data["atr_pct"] = data["atr_14"] / data["close"].replace(0, np.nan)
    rolling_mean_20 = data["close"].rolling(20).mean()
    rolling_std_20 = data["close"].rolling(20).std()
    data["close_zscore_20"] = (data["close"] - rolling_mean_20) / rolling_std_20.replace(0, np.nan)
    data["range_pct"] = (data["high"] - data["low"]) / data["close"].replace(0, np.nan)
    data["body_pct"] = (data["close"] - data["open"]).abs() / data["close"].replace(0, np.nan)
    candle_top = data[["open", "close"]].max(axis=1)
    candle_bottom = data[["open", "close"]].min(axis=1)
    data["upper_wick_pct"] = (data["high"] - candle_top).clip(lower=0) / data["close"].replace(0, np.nan)
    data["lower_wick_pct"] = (candle_bottom - data["low"]).clip(lower=0) / data["close"].replace(0, np.nan)
    data["range_to_atr"] = (data["high"] - data["low"]) / data["atr_14"].replace(0, np.nan)
    rolling_high_20 = data["high"].rolling(20).max()
    rolling_low_20 = data["low"].rolling(20).min()
    data["roll_high_gap_20"] = data["close"] / rolling_high_20.replace(0, np.nan) - 1.0
    data["roll_low_gap_20"] = data["close"] / rolling_low_20.replace(0, np.nan) - 1.0
    data["donchian_width_20"] = (rolling_high_20 - rolling_low_20) / data["close"].replace(0, np.nan)
    data["spread_pct"] = data["spread"] / data["close"].replace(0, np.nan) if "spread" in data.columns else 0.0
    if "tick_volume" in data.columns:
        volume_mean_20 = data["tick_volume"].rolling(20).mean()
        volume_std_20 = data["tick_volume"].rolling(20).std()
        data["volume_change_1"] = data["tick_volume"].pct_change(1)
        data["volume_zscore_20"] = (data["tick_volume"] - volume_mean_20) / volume_std_20.replace(0, np.nan)
        obv_direction = np.sign(data["close"].diff().fillna(0.0))
        obv = (obv_direction * data["tick_volume"]).cumsum()
        obv_mean = obv.rolling(20).mean()
        obv_std = obv.rolling(20).std()
        data["obv_zscore_20"] = (obv - obv_mean) / obv_std.replace(0, np.nan)
        data["volume_price_trend"] = ((data["close"].pct_change().fillna(0.0)) * data["tick_volume"]).cumsum()
    else:
        data["volume_change_1"] = 0.0
        data["volume_zscore_20"] = 0.0
        data["obv_zscore_20"] = 0.0
        data["volume_price_trend"] = 0.0
    data["roc_10"] = data["close"].pct_change(10)
    data["atr_accel_3"] = data["atr_14"].pct_change(3)

    hour = data["time"].dt.hour
    day_of_week = data["time"].dt.dayofweek
    data["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    data["dow_sin"] = np.sin(2 * math.pi * day_of_week / 7)
    data["dow_cos"] = np.cos(2 * math.pi * day_of_week / 7)

    data["session_tokyo"] = hour.between(0, 8, inclusive="left").astype(int)
    data["session_london"] = hour.between(7, 16, inclusive="left").astype(int)
    data["session_new_york"] = hour.between(13, 22, inclusive="left").astype(int)
    data["session_overlap"] = ((data["session_london"] == 1) & (data["session_new_york"] == 1)).astype(int)

    data["btc_eth_corr_48"] = 0.0
    data["btc_dxy_corr_48"] = 0.0
    if symbol_upper in {"BTCUSD", "ETHUSD"}:
        counterpart = "ETHUSD" if symbol_upper == "BTCUSD" else "BTCUSD"
        counterpart_frame = refs.get(counterpart)
        if counterpart_frame is not None and not counterpart_frame.empty:
            counterpart_close = _aligned_series_by_time(data["time"], counterpart_frame, "close")
            counterpart_returns = counterpart_close.pct_change(1)
            data["btc_eth_corr_48"] = data["return_1"].rolling(48).corr(counterpart_returns)

    dxy_frame = refs.get("DXY")
    btc_frame = refs.get("BTCUSD")
    if dxy_frame is not None and not dxy_frame.empty:
        dxy_close = _aligned_series_by_time(data["time"], dxy_frame, "close")
        dxy_returns = dxy_close.pct_change(1)
        if symbol_upper == "BTCUSD":
            btc_returns = data["return_1"]
        elif symbol_upper == "ETHUSD" and btc_frame is not None and not btc_frame.empty:
            btc_close = _aligned_series_by_time(data["time"], btc_frame, "close")
            btc_returns = btc_close.pct_change(1)
        else:
            btc_returns = pd.Series(np.nan, index=data.index)
        data["btc_dxy_corr_48"] = btc_returns.rolling(48).corr(dxy_returns)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.drop(columns=["bb_upper", "bb_lower"], inplace=True, errors="ignore")
    return data


def build_supervised_frame(
    frame: pd.DataFrame,
    horizon: int,
    edge_bps: int,
    breakout_pct: float | None = None,
    symbol: str | None = None,
    reference_frames: dict[str, pd.DataFrame] | None = None,
    funding_rate: float | None = None,
) -> pd.DataFrame:
    features = build_feature_frame(
        frame,
        symbol=symbol,
        reference_frames=reference_frames,
        funding_rate=funding_rate,
    )
    features["future_return"] = features["close"].shift(-horizon) / features["close"] - 1.0
    features["target_long"] = 0
    features["target_short"] = 0
    if breakout_pct is not None and breakout_pct > 0:
        future_high = features["high"][::-1].rolling(window=horizon, min_periods=horizon).max()[::-1].shift(-1)
        future_low = features["low"][::-1].rolling(window=horizon, min_periods=horizon).min()[::-1].shift(-1)
        up_breakout = (future_high / features["close"] - 1.0) >= breakout_pct
        down_breakout = (future_low / features["close"] - 1.0) <= -breakout_pct
        features.loc[up_breakout, "target_long"] = 1
        features.loc[down_breakout, "target_short"] = 1
    else:
        edge = edge_bps / 10_000.0
        features.loc[features["future_return"] > edge, "target_long"] = 1
        features.loc[features["future_return"] < -edge, "target_short"] = 1
    return features.dropna(subset=FEATURE_COLUMNS + ["future_return"]).reset_index(drop=True)


def build_training_frame(
    frame: pd.DataFrame,
    horizon: int,
    edge_bps: int,
    target_side: str = "long",
    breakout_pct: float | None = None,
    symbol: str | None = None,
    reference_frames: dict[str, pd.DataFrame] | None = None,
    funding_rate: float | None = None,
) -> pd.DataFrame:
    features = build_supervised_frame(
        frame,
        horizon=horizon,
        edge_bps=edge_bps,
        breakout_pct=breakout_pct,
        symbol=symbol,
        reference_frames=reference_frames,
        funding_rate=funding_rate,
    )
    side = target_side.lower()
    if side == "long":
        features["target"] = features["target_long"]
    elif side == "short":
        features["target"] = features["target_short"]
    else:
        raise ValueError(f"Unsupported target_side: {target_side}")
    return features.dropna(subset=FEATURE_COLUMNS + ["future_return", "target"]).reset_index(drop=True)


def select_feature_columns(
    frame: pd.DataFrame,
    target_column: str,
    min_features: int = 12,
    max_features: int = 24,
    forced_features: list[str] | None = None,
    correlation_threshold: float = 0.95,
    min_mutual_info: float = 0.0,
) -> list[str]:
    forced = [feature for feature in (forced_features or []) if feature in FEATURE_COLUMNS]
    if target_column not in frame.columns:
        defaults = [feature for feature in BASE_FEATURE_COLUMNS if feature in frame.columns]
        return list(dict.fromkeys([*forced, *defaults[: max(min_features, 1)]]))

    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError:
        defaults = [feature for feature in BASE_FEATURE_COLUMNS if feature in frame.columns]
        return list(dict.fromkeys([*forced, *defaults[: max(min_features, 1)]]))

    usable_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    candidate_frame = frame[usable_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if candidate_frame.empty:
        defaults = [feature for feature in BASE_FEATURE_COLUMNS if feature in frame.columns]
        return list(dict.fromkeys([*forced, *defaults[: max(min_features, 1)]]))
    target = frame.loc[candidate_frame.index, target_column].astype(int)
    if target.nunique() < 2:
        defaults = [feature for feature in BASE_FEATURE_COLUMNS if feature in candidate_frame.columns]
        return list(dict.fromkeys([*forced, *defaults[: max(min_features, 1)]]))

    scores = mutual_info_classif(candidate_frame, target, discrete_features=False, random_state=42)
    ranking = (
        pd.DataFrame({"feature": candidate_frame.columns, "score": scores})
        .sort_values(["score", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    chosen = [feature for feature in forced if feature in candidate_frame.columns]
    for feature in BASE_FEATURE_COLUMNS:
        if feature in candidate_frame.columns and feature not in chosen:
            chosen.append(feature)
        if len(chosen) >= min_features:
            break

    correlation_frame = candidate_frame.corr().abs()
    for row in ranking.itertuples(index=False):
        feature = str(row.feature)
        score = float(row.score)
        if score < float(min_mutual_info) or feature in chosen:
            continue
        highly_correlated = any(
            float(correlation_frame.loc[feature, existing]) >= float(correlation_threshold)
            for existing in chosen
            if feature in correlation_frame.index and existing in correlation_frame.columns
        )
        if highly_correlated:
            continue
        chosen.append(feature)
        if len(chosen) >= max_features:
            break

    if len(chosen) < min_features:
        for feature in ranking["feature"].tolist():
            if feature not in chosen:
                chosen.append(feature)
            if len(chosen) >= min_features:
                break
    return chosen[: max_features]

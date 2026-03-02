from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import Settings
from modules.types import TradeSignal


class SignalEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _is_crypto_symbol(self, symbol: str) -> bool:
        return symbol.upper() in {item.upper() for item in self.settings.trading.crypto_symbols}

    def _allowed_sessions_for_symbol(self, symbol: str) -> list[str]:
        mapped = self.settings.trading.symbol_allowed_sessions.get(symbol.upper())
        if mapped is None:
            return [str(session).lower() for session in self.settings.trading.allowed_sessions]
        normalized = [str(session).lower() for session in mapped]
        if "all" in normalized:
            return []
        return normalized

    def _allowed_hours_for_symbol(self, symbol: str) -> list[int]:
        mapped = self.settings.trading.symbol_allowed_hours_utc.get(symbol.upper())
        if mapped:
            return mapped
        return self.settings.trading.allowed_hours_utc

    def _session_allowed(self, symbol: str, row: pd.Series) -> bool:
        allowed_sessions = self._allowed_sessions_for_symbol(symbol)
        if not allowed_sessions:
            return True
        mapping = {
            "tokyo": bool(row.get("session_tokyo", 0)),
            "london": bool(row.get("session_london", 0)),
            "new_york": bool(row.get("session_new_york", 0)),
            "overlap": bool(row.get("session_overlap", 0)),
        }
        return any(mapping.get(session.lower(), False) for session in allowed_sessions)

    def _long_blockers(
        self,
        symbol: str,
        row: pd.Series,
        long_probability: float,
        short_probability: float,
        long_threshold: float,
    ) -> list[str]:
        blockers = []
        is_crypto = self._is_crypto_symbol(symbol)
        if long_probability < long_threshold:
            blockers.append("prob_below_threshold")
        if long_probability < short_probability:
            blockers.append("short_dominates")
        
        dominance_margin = self.settings.dominance_margin_for(symbol)
        if (long_probability - short_probability) < dominance_margin:
            blockers.append("insufficient_dominance")
        
        if not is_crypto and row["ema_fast"] <= row["ema_slow"]:
            blockers.append("ema_bearish")
        if is_crypto:
            if not (20 <= row["rsi_14"] <= 85):
                blockers.append("rsi_out_of_range")
        elif not (45 <= row["rsi_14"] <= 72):
            blockers.append("rsi_out_of_range")
        if row["atr_14"] <= 0:
            blockers.append("atr_zero")
        allowed_sessions = self._allowed_sessions_for_symbol(symbol)
        if allowed_sessions:
            in_session = any(
                row.get(f"session_{session}", False) for session in allowed_sessions
            )
            if not in_session:
                blockers.append("session_filter")
        
        if hasattr(row, "name") and hasattr(row.name, "hour"):
            current_hour = row.name.hour
        elif "time" in row.index and hasattr(row["time"], "hour"):
            current_hour = row["time"].hour
        else:
            current_hour = datetime.now(timezone.utc).hour
        
        allowed_hours = self._allowed_hours_for_symbol(symbol)
        if allowed_hours and current_hour not in allowed_hours:
            blockers.append("outside_allowed_hours")
        
        return blockers

    def _short_blockers(
        self,
        symbol: str,
        row: pd.Series,
        long_probability: float,
        short_probability: float,
        short_threshold: float,
    ) -> list[str]:
        blockers = []
        is_crypto = self._is_crypto_symbol(symbol)
        if short_probability < short_threshold:
            blockers.append("prob_below_threshold")
        if short_probability < long_probability:
            blockers.append("long_dominates")
        
        dominance_margin = self.settings.dominance_margin_for(symbol)
        if (short_probability - long_probability) < dominance_margin:
            blockers.append("insufficient_dominance")
        
        if not is_crypto and row["ema_fast"] >= row["ema_slow"]:
            blockers.append("ema_bullish")
        if is_crypto:
            if not (15 <= row["rsi_14"] <= 80):
                blockers.append("rsi_out_of_range")
        elif not (28 <= row["rsi_14"] <= 55):
            blockers.append("rsi_out_of_range")
        if row["atr_14"] <= 0:
            blockers.append("atr_zero")
        allowed_sessions = self._allowed_sessions_for_symbol(symbol)
        if allowed_sessions:
            in_session = any(
                row.get(f"session_{session}", False) for session in allowed_sessions
            )
            if not in_session:
                blockers.append("session_filter")
        
        if hasattr(row, "name") and hasattr(row.name, "hour"):
            current_hour = row.name.hour
        elif "time" in row.index and hasattr(row["time"], "hour"):
            current_hour = row["time"].hour
        else:
            current_hour = datetime.now(timezone.utc).hour
        
        allowed_hours = self._allowed_hours_for_symbol(symbol)
        if allowed_hours and current_hour not in allowed_hours:
            blockers.append("outside_allowed_hours")
        
        return blockers

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        feature_frame: pd.DataFrame,
        long_probability: float,
        short_probability: float,
        chronos_direction: float | None,
        long_threshold: float | None = None,
        short_threshold: float | None = None,
    ) -> TradeSignal:
        row = feature_frame.iloc[-1]
        chronos_deadband = self.settings.model.chronos_deadband
        long_threshold = (
            float(long_threshold)
            if long_threshold is not None
            else self.settings.threshold_for(symbol, "long")
        )
        short_threshold = (
            float(short_threshold)
            if short_threshold is not None
            else self.settings.threshold_for(symbol, "short")
        )

        side = "flat"
        reason = "no_edge"
        selected_probability = max(long_probability, short_probability)
        long_blockers = self._long_blockers(symbol, row, long_probability, short_probability, long_threshold)
        short_blockers = self._short_blockers(symbol, row, long_probability, short_probability, short_threshold)

        if (
            self.settings.trading.allow_long
            and long_probability >= long_threshold
            and not long_blockers
            and (chronos_direction is None or chronos_direction >= -chronos_deadband)
        ):
            side = "long"
            reason = "xgb_long_filter"
        elif (
            self.settings.trading.allow_short
            and short_probability >= short_threshold
            and not short_blockers
            and (chronos_direction is None or chronos_direction <= chronos_deadband)
        ):
            side = "short"
            reason = "xgb_short_filter"
        else:
            dominant = "long" if long_probability >= short_probability else "short"
            blockers = long_blockers if dominant == "long" else short_blockers
            if chronos_direction is not None:
                if dominant == "long" and chronos_direction < -chronos_deadband:
                    blockers = [*blockers, "chronos_blocks_long"]
                if dominant == "short" and chronos_direction > chronos_deadband:
                    blockers = [*blockers, "chronos_blocks_short"]
            reason = ",".join(blockers[:4]) if blockers else "no_edge"

        score = float(
            min(
                1.0,
                max(abs(long_probability - 0.5), abs(short_probability - 0.5)) * 2
                + abs(chronos_direction or 0.0) * 20,
            )
        )
        features = {
            "close": float(row["close"]),
            "ema_fast": float(row["ema_fast"]),
            "ema_slow": float(row["ema_slow"]),
            "rsi_14": float(row["rsi_14"]),
            "atr_14": float(row["atr_14"]),
        }
        return TradeSignal(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            probability=float(selected_probability),
            long_probability=float(long_probability),
            short_probability=float(short_probability),
            score=score,
            reason=reason,
            generated_at=datetime.now(timezone.utc),
            chronos_direction=chronos_direction,
            features=features,
        )

    def vectorized_signals(
        self,
        symbol: str,
        feature_frame: pd.DataFrame,
        long_probabilities: pd.Series,
        short_probabilities: pd.Series,
        chronos_series: pd.Series | None = None,
        long_threshold: float | None = None,
        short_threshold: float | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        long_threshold = (
            float(long_threshold)
            if long_threshold is not None
            else self.settings.threshold_for(symbol, "long")
        )
        short_threshold = (
            float(short_threshold)
            if short_threshold is not None
            else self.settings.threshold_for(symbol, "short")
        )
        is_crypto = self._is_crypto_symbol(symbol)
        chronos_deadband = self.settings.model.chronos_deadband
        dominance_margin = self.settings.dominance_margin_for(symbol)
        long_probs = long_probabilities.reindex(feature_frame.index).fillna(0.5)
        short_probs = short_probabilities.reindex(feature_frame.index).fillna(0.5)

        allowed_sessions = self._allowed_sessions_for_symbol(symbol)
        session_mask = pd.Series(True, index=feature_frame.index)
        if allowed_sessions:
            session_mask = pd.Series(False, index=feature_frame.index)
            for session in allowed_sessions:
                key = {
                    "tokyo": "session_tokyo",
                    "london": "session_london",
                    "new_york": "session_new_york",
                    "overlap": "session_overlap",
                }.get(session.lower())
                if key:
                    session_mask |= feature_frame[key].astype(bool)

        allowed_hours = self._allowed_hours_for_symbol(symbol)
        if isinstance(feature_frame.index, pd.DatetimeIndex):
            hours = feature_frame.index.hour
        elif "time" in feature_frame.columns:
            hours = pd.to_datetime(feature_frame["time"], utc=True).dt.hour
        else:
            hours = pd.Series(datetime.now(timezone.utc).hour, index=feature_frame.index)
        hours_mask = pd.Series(True, index=feature_frame.index)
        if allowed_hours:
            hours_mask = pd.Series(hours, index=feature_frame.index).isin(allowed_hours)

        long_trend_filter = pd.Series(True, index=feature_frame.index)
        short_trend_filter = pd.Series(True, index=feature_frame.index)
        if not is_crypto:
            long_trend_filter = feature_frame["ema_fast"] > feature_frame["ema_slow"]
            short_trend_filter = feature_frame["ema_fast"] < feature_frame["ema_slow"]

        long_rsi_filter = feature_frame["rsi_14"].between(45, 72)
        short_rsi_filter = feature_frame["rsi_14"].between(28, 55)
        if is_crypto:
            long_rsi_filter = feature_frame["rsi_14"].between(20, 85)
            short_rsi_filter = feature_frame["rsi_14"].between(15, 80)

        long_entries = (
            (long_probs >= long_threshold)
            & ((long_probs - short_probs) >= dominance_margin)
            & long_trend_filter
            & long_rsi_filter
            & (feature_frame["atr_14"] > 0)
            & session_mask
            & hours_mask
        )
        short_entries = (
            (short_probs >= short_threshold)
            & ((short_probs - long_probs) >= dominance_margin)
            & short_trend_filter
            & short_rsi_filter
            & (feature_frame["atr_14"] > 0)
            & session_mask
            & hours_mask
        )

        if chronos_series is not None:
            chronos = chronos_series.reindex(feature_frame.index).fillna(0.0)
            long_entries &= chronos >= -chronos_deadband
            short_entries &= chronos <= chronos_deadband

        if not self.settings.trading.allow_long:
            long_entries = pd.Series(False, index=feature_frame.index)
        if not self.settings.trading.allow_short:
            short_entries = pd.Series(False, index=feature_frame.index)

        return long_entries.fillna(False), short_entries.fillna(False)


def heuristic_probabilities(row: pd.Series) -> tuple[float, float]:
    ema_bias = np.tanh(float(row["ema_gap_pct"]) * 400.0) * 0.18
    rsi_bias = ((float(row["rsi_14"]) - 50.0) / 50.0) * 0.10
    directional_bias = ema_bias + rsi_bias
    long_probability = float(np.clip(0.5 + directional_bias, 0.01, 0.99))
    short_probability = float(np.clip(0.5 - directional_bias, 0.01, 0.99))
    return long_probability, short_probability

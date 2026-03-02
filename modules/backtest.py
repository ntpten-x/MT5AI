from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import Settings
from modules.brain_ai import LightGBMSignalModel, XGBoostSignalModel
from modules.feature_engineering import build_supervised_frame
from modules.signals import SignalEngine
from modules.timeframes import pandas_frequency


class BacktestEngine:
    def __init__(self, settings: Settings, signal_engine: SignalEngine):
        self.settings = settings
        self.signal_engine = signal_engine

    def _require_vectorbt(self):
        try:
            import vectorbt as vbt
        except ImportError as exc:
            raise RuntimeError("VectorBT is not installed") from exc
        return vbt

    def _heuristic_probabilities(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        ema_bias = np.tanh(frame["ema_gap_pct"].fillna(0.0) * 400.0) * 0.18
        rsi_bias = ((frame["rsi_14"].fillna(50.0) - 50.0) / 50.0) * 0.10
        directional_bias = ema_bias + rsi_bias
        long_probabilities = (0.5 + directional_bias).clip(0.01, 0.99)
        short_probabilities = (0.5 - directional_bias).clip(0.01, 0.99)
        return long_probabilities, short_probabilities

    def run_from_probabilities(
        self,
        symbol: str,
        feature_frame: pd.DataFrame,
        long_probabilities: pd.Series,
        short_probabilities: pd.Series,
        chronos_series: pd.Series | None = None,
        long_threshold: float | None = None,
        short_threshold: float | None = None,
        risk_per_trade: float | None = None,
        atr_stop_loss_mult: float | None = None,
        atr_take_profit_mult: float | None = None,
    ) -> dict[str, Any]:
        vbt = self._require_vectorbt()
        long_entries, short_entries = self.signal_engine.vectorized_signals(
            symbol,
            feature_frame,
            long_probabilities,
            short_probabilities,
            chronos_series=chronos_series,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
        )
        risk_fraction = float(risk_per_trade) if risk_per_trade is not None else self.settings.risk_per_trade_for(symbol)
        stop_mult = (
            float(atr_stop_loss_mult)
            if atr_stop_loss_mult is not None
            else self.settings.atr_stop_loss_mult_for(symbol)
        )
        take_profit_mult = (
            float(atr_take_profit_mult)
            if atr_take_profit_mult is not None
            else self.settings.atr_take_profit_mult_for(symbol)
        )
        close = feature_frame["close"]
        atr_stop_pct = ((feature_frame["atr_14"] * stop_mult) / close.replace(0, np.nan)).clip(lower=0.0005)
        atr_tp_pct = ((feature_frame["atr_14"] * take_profit_mult) / close.replace(0, np.nan)).clip(lower=0.0008)
        size = (risk_fraction / atr_stop_pct.replace(0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0)
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=long_entries,
            exits=short_entries,
            short_entries=short_entries,
            short_exits=long_entries,
            size=size,
            size_type=vbt.portfolio.enums.SizeType.Percent,
            fees=self.settings.execution.fee_bps / 10_000.0,
            slippage=self.settings.execution.slippage_bps / 10_000.0,
            sl_stop=atr_stop_pct.fillna(0.0),
            tp_stop=atr_tp_pct.fillna(0.0),
            freq=pandas_frequency(self.settings.trading.timeframe),
        )
        stats = {str(key): _serialize(value) for key, value in portfolio.stats().to_dict().items()}
        stats["long_signal_count"] = int(long_entries.sum())
        stats["short_signal_count"] = int(short_entries.sum())
        stats["risk_per_trade"] = risk_fraction
        stats["atr_stop_loss_mult"] = stop_mult
        stats["atr_take_profit_mult"] = take_profit_mult
        return stats

    def _blend_probability_streams(
        self,
        long_probabilities: pd.Series,
        short_probabilities: pd.Series,
        lightgbm_long_probabilities: pd.Series | None = None,
        lightgbm_short_probabilities: pd.Series | None = None,
        lightgbm_weight: float = 0.0,
        sequence_long_probabilities: pd.Series | None = None,
        sequence_short_probabilities: pd.Series | None = None,
        sequence_weight: float = 0.0,
        transformer_long_probabilities: pd.Series | None = None,
        transformer_short_probabilities: pd.Series | None = None,
        transformer_weight: float = 0.0,
    ) -> tuple[pd.Series, pd.Series]:
        long_stream = long_probabilities.astype(float).clip(0.01, 0.99)
        short_stream = short_probabilities.astype(float).clip(0.01, 0.99)
        auxiliaries: list[tuple[pd.Series | None, pd.Series | None, float]] = [
            (lightgbm_long_probabilities, lightgbm_short_probabilities, float(lightgbm_weight)),
            (sequence_long_probabilities, sequence_short_probabilities, float(sequence_weight)),
            (transformer_long_probabilities, transformer_short_probabilities, float(transformer_weight)),
        ]
        total_weight = sum(max(0.0, weight) for _, _, weight in auxiliaries)
        if total_weight <= 0:
            return long_stream, short_stream
        normalized_total = min(total_weight, 0.8)
        if normalized_total <= 0:
            return long_stream, short_stream
        blended_long = long_stream.copy()
        blended_short = short_stream.copy()
        base_scale = max(0.0, 1.0 - normalized_total)
        blended_long = blended_long * base_scale
        blended_short = blended_short * base_scale
        for aux_long, aux_short, raw_weight in auxiliaries:
            if aux_long is None or aux_short is None or raw_weight <= 0:
                continue
            normalized_weight = normalized_total * (raw_weight / total_weight)
            aligned_long = aux_long.reindex(long_stream.index).fillna(0.5).astype(float).clip(0.01, 0.99)
            aligned_short = aux_short.reindex(short_stream.index).fillna(0.5).astype(float).clip(0.01, 0.99)
            blended_long = blended_long + (normalized_weight * aligned_long)
            blended_short = blended_short + (normalized_weight * aligned_short)
        return blended_long.clip(0.01, 0.99), blended_short.clip(0.01, 0.99)

    def optimize_parameters(
        self,
        symbol: str,
        feature_frame: pd.DataFrame,
        long_probabilities: pd.Series,
        short_probabilities: pd.Series,
        sequence_long_probabilities: pd.Series | None = None,
        sequence_short_probabilities: pd.Series | None = None,
        lightgbm_long_probabilities: pd.Series | None = None,
        lightgbm_short_probabilities: pd.Series | None = None,
        transformer_long_probabilities: pd.Series | None = None,
        transformer_short_probabilities: pd.Series | None = None,
        chronos_series: pd.Series | None = None,
    ) -> dict[str, Any]:
        has_sequence = sequence_long_probabilities is not None and sequence_short_probabilities is not None
        has_lightgbm = lightgbm_long_probabilities is not None and lightgbm_short_probabilities is not None
        has_transformer = (
            transformer_long_probabilities is not None and transformer_short_probabilities is not None
        )
        if self.settings.model.optuna_enabled:
            try:
                import optuna
            except ImportError:
                optuna = None
        else:
            optuna = None

        def _evaluate_candidate(
            long_threshold: float,
            short_threshold: float,
            risk_per_trade: float,
            atr_stop_loss_mult: float,
            atr_take_profit_mult: float,
            lightgbm_weight: float,
            sequence_weight: float,
            transformer_weight: float,
        ) -> dict[str, Any]:
            blended_long, blended_short = self._blend_probability_streams(
                long_probabilities,
                short_probabilities,
                lightgbm_long_probabilities=lightgbm_long_probabilities,
                lightgbm_short_probabilities=lightgbm_short_probabilities,
                lightgbm_weight=lightgbm_weight,
                sequence_long_probabilities=sequence_long_probabilities,
                sequence_short_probabilities=sequence_short_probabilities,
                sequence_weight=sequence_weight,
                transformer_long_probabilities=transformer_long_probabilities,
                transformer_short_probabilities=transformer_short_probabilities,
                transformer_weight=transformer_weight,
            )
            try:
                stats = self.run_from_probabilities(
                    symbol,
                    feature_frame,
                    blended_long,
                    blended_short,
                    chronos_series=chronos_series,
                    long_threshold=long_threshold,
                    short_threshold=short_threshold,
                    risk_per_trade=risk_per_trade,
                    atr_stop_loss_mult=atr_stop_loss_mult,
                    atr_take_profit_mult=atr_take_profit_mult,
                )
            except ValueError as exc:
                if "position reversal" not in str(exc).lower():
                    raise
                stats = {
                    "Total Return [%]": -50.0,
                    "Max Drawdown [%]": 50.0,
                    "Total Trades": 0,
                    "Win Rate [%]": 0.0,
                    "Sharpe Ratio": -5.0,
                    "optimization_error": str(exc),
                }
            objective = self._objective(stats)
            return {
                "long_threshold": float(long_threshold),
                "short_threshold": float(short_threshold),
                "risk_per_trade": float(risk_per_trade),
                "atr_stop_loss_mult": float(atr_stop_loss_mult),
                "atr_take_profit_mult": float(atr_take_profit_mult),
                "lightgbm_probability_weight": float(lightgbm_weight),
                "sequence_probability_weight": float(sequence_weight),
                "transformer_probability_weight": float(transformer_weight),
                "objective": float(objective),
                "stats": stats,
            }

        best_result: dict[str, Any] | None = None
        if optuna is not None:
            sampler = optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=max(1, self.settings.model.optuna_n_startup_trials),
            )
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def _objective_fn(trial):
                candidate = _evaluate_candidate(
                    long_threshold=trial.suggest_float("long_threshold", 0.50, 0.80, step=0.01),
                    short_threshold=trial.suggest_float("short_threshold", 0.50, 0.80, step=0.01),
                    risk_per_trade=trial.suggest_float(
                        "risk_per_trade",
                        self.settings.model.optuna_risk_min,
                        self.settings.model.optuna_risk_max,
                        step=0.0005,
                    ),
                    atr_stop_loss_mult=trial.suggest_float(
                        "atr_stop_loss_mult",
                        self.settings.model.optuna_sl_min,
                        self.settings.model.optuna_sl_max,
                        step=0.1,
                    ),
                    atr_take_profit_mult=trial.suggest_float(
                        "atr_take_profit_mult",
                        self.settings.model.optuna_tp_min,
                        self.settings.model.optuna_tp_max,
                        step=0.1,
                    ),
                    lightgbm_weight=trial.suggest_float(
                        "lightgbm_probability_weight",
                        0.0,
                        self.settings.model.optuna_lightgbm_weight_max if has_lightgbm else 0.0,
                        step=0.05,
                    )
                    if has_lightgbm
                    else 0.0,
                    sequence_weight=trial.suggest_float(
                        "sequence_probability_weight",
                        0.0,
                        self.settings.model.optuna_sequence_weight_max if has_sequence else 0.0,
                        step=0.05,
                    )
                    if has_sequence
                    else 0.0,
                    transformer_weight=trial.suggest_float(
                        "transformer_probability_weight",
                        0.0,
                        self.settings.model.optuna_transformer_weight_max if has_transformer else 0.0,
                        step=0.05,
                    )
                    if has_transformer
                    else 0.0,
                )
                trial.set_user_attr("candidate", candidate)
                return float(candidate["objective"])

            study.optimize(
                _objective_fn,
                n_trials=max(1, self.settings.model.optuna_trials),
                timeout=max(1, self.settings.model.optuna_timeout_seconds),
                show_progress_bar=False,
            )
            best_trial = study.best_trial
            best_result = dict(best_trial.user_attrs["candidate"])
            best_result["study"] = {
                "best_value": float(study.best_value),
                "trial_count": len(study.trials),
            }
        else:
            candidates = sorted({float(value) for value in self.settings.model.xgb_threshold_candidates})
            risk_candidates = sorted(
                {
                    float(self.settings.risk_per_trade_for(symbol)),
                    float(self.settings.model.optuna_risk_min),
                    float(self.settings.model.optuna_risk_max),
                }
            )
            stop_candidates = sorted(
                {
                    float(self.settings.atr_stop_loss_mult_for(symbol)),
                    float(self.settings.model.optuna_sl_min),
                    float(self.settings.model.optuna_sl_max),
                }
            )
            tp_candidates = sorted(
                {
                    float(self.settings.atr_take_profit_mult_for(symbol)),
                    float(self.settings.model.optuna_tp_min),
                    float(self.settings.model.optuna_tp_max),
                }
            )
            sequence_candidates = (
                [0.0, float(self.settings.sequence_probability_weight_for(symbol))]
                if has_sequence
                else [0.0]
            )
            lightgbm_candidates = (
                [0.0, float(self.settings.lightgbm_probability_weight_for(symbol))]
                if has_lightgbm
                else [0.0]
            )
            transformer_candidates = (
                [0.0, float(self.settings.transformer_probability_weight_for(symbol))]
                if has_transformer
                else [0.0]
            )
            for long_threshold in candidates:
                for short_threshold in candidates:
                    for risk_per_trade in risk_candidates:
                        for atr_stop_loss_mult in stop_candidates:
                            for atr_take_profit_mult in tp_candidates:
                                for lightgbm_weight in lightgbm_candidates:
                                    for sequence_weight in sequence_candidates:
                                        for transformer_weight in transformer_candidates:
                                            candidate = _evaluate_candidate(
                                                long_threshold=long_threshold,
                                                short_threshold=short_threshold,
                                                risk_per_trade=risk_per_trade,
                                                atr_stop_loss_mult=atr_stop_loss_mult,
                                                atr_take_profit_mult=atr_take_profit_mult,
                                                lightgbm_weight=lightgbm_weight,
                                                sequence_weight=sequence_weight,
                                                transformer_weight=transformer_weight,
                                            )
                                            if (
                                                best_result is None
                                                or candidate["objective"] > best_result["objective"]
                                            ):
                                                best_result = candidate

        if best_result is None:
            raise RuntimeError("Parameter optimization failed")
        return best_result

    def optimize_thresholds(
        self,
        symbol: str,
        feature_frame: pd.DataFrame,
        long_probabilities: pd.Series,
        short_probabilities: pd.Series,
    ) -> dict[str, Any]:
        tuned = self.optimize_parameters(symbol, feature_frame, long_probabilities, short_probabilities)
        return {
            "long_threshold": tuned["long_threshold"],
            "short_threshold": tuned["short_threshold"],
            "objective": tuned["objective"],
            "stats": tuned["stats"],
        }

    def run(
        self,
        symbol: str,
        market_frame: pd.DataFrame,
        long_model: XGBoostSignalModel | None = None,
        short_model: XGBoostSignalModel | None = None,
        lgbm_long_model: LightGBMSignalModel | None = None,
        lgbm_short_model: LightGBMSignalModel | None = None,
        sequence_long_model: Any | None = None,
        sequence_short_model: Any | None = None,
        transformer_long_model: Any | None = None,
        transformer_short_model: Any | None = None,
    ) -> dict[str, Any]:
        feature_frame = build_supervised_frame(
            market_frame,
            horizon=self.settings.model.positive_return_horizon,
            edge_bps=self.settings.model.positive_return_bps,
            breakout_pct=self.settings.model.volatility_breakout_pct,
            symbol=symbol,
        )
        if feature_frame.empty:
            raise ValueError("Not enough data for backtest")

        if (
            long_model is not None
            and short_model is not None
            and long_model.is_ready()
            and short_model.is_ready()
        ):
            long_probabilities = long_model.predict_proba_frame(feature_frame)
            short_probabilities = short_model.predict_proba_frame(feature_frame)
            long_threshold = long_model.effective_threshold(self.settings.threshold_for(symbol, "long"))
            short_threshold = short_model.effective_threshold(self.settings.threshold_for(symbol, "short"))
        else:
            long_probabilities, short_probabilities = self._heuristic_probabilities(feature_frame)
            long_threshold = self.settings.threshold_for(symbol, "long")
            short_threshold = self.settings.threshold_for(symbol, "short")
        lightgbm_weight = self.settings.lightgbm_probability_weight_for(symbol)
        sequence_weight = self.settings.sequence_probability_weight_for(symbol)
        transformer_weight = self.settings.transformer_probability_weight_for(symbol)
        has_lightgbm_models = (
            lgbm_long_model is not None
            and lgbm_short_model is not None
            and lgbm_long_model.is_ready()
            and lgbm_short_model.is_ready()
        )
        has_sequence_models = (
            sequence_long_model is not None
            and sequence_short_model is not None
            and sequence_long_model.is_ready()
            and sequence_short_model.is_ready()
        )
        has_transformer_models = (
            transformer_long_model is not None
            and transformer_short_model is not None
            and transformer_long_model.is_ready()
            and transformer_short_model.is_ready()
        )
        if has_lightgbm_models or has_sequence_models or has_transformer_models:
            long_probabilities, short_probabilities = self._blend_probability_streams(
                long_probabilities,
                short_probabilities,
                lightgbm_long_probabilities=(
                    lgbm_long_model.predict_proba_frame(feature_frame) if has_lightgbm_models else None
                ),
                lightgbm_short_probabilities=(
                    lgbm_short_model.predict_proba_frame(feature_frame) if has_lightgbm_models else None
                ),
                lightgbm_weight=lightgbm_weight,
                sequence_long_probabilities=(
                    sequence_long_model.predict_proba_frame(feature_frame) if has_sequence_models else None
                ),
                sequence_short_probabilities=(
                    sequence_short_model.predict_proba_frame(feature_frame) if has_sequence_models else None
                ),
                sequence_weight=sequence_weight,
                transformer_long_probabilities=(
                    transformer_long_model.predict_proba_frame(feature_frame) if has_transformer_models else None
                ),
                transformer_short_probabilities=(
                    transformer_short_model.predict_proba_frame(feature_frame) if has_transformer_models else None
                ),
                transformer_weight=transformer_weight,
            )
        return self.run_from_probabilities(
            symbol,
            feature_frame,
            long_probabilities,
            short_probabilities,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            risk_per_trade=self.settings.risk_per_trade_for(symbol),
            atr_stop_loss_mult=self.settings.atr_stop_loss_mult_for(symbol),
            atr_take_profit_mult=self.settings.atr_take_profit_mult_for(symbol),
        )

    def _objective(self, stats: dict[str, Any]) -> float:
        total_return = float(stats.get("Total Return [%]") or 0.0)
        max_drawdown = float(stats.get("Max Drawdown [%]") or 0.0)
        total_trades = int(stats.get("Total Trades") or 0)
        win_rate = float(stats.get("Win Rate [%]") or 0.0)
        sharpe = stats.get("Sharpe Ratio")
        if sharpe in (None, np.inf, -np.inf):
            sharpe_value = 0.0
        else:
            sharpe_value = float(sharpe)

        score = total_return - (0.8 * max_drawdown) + (0.5 * max(min(sharpe_value, 5.0), -5.0))
        score += 0.02 * win_rate
        if total_trades < self.settings.model.tuning_min_trades:
            score -= 5.0 * (self.settings.model.tuning_min_trades - total_trades + 1)
        return score


def _serialize(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value

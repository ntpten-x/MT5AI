from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

from config import Settings
from modules.backtest import BacktestEngine
from modules.brain_ai import LightGBMSignalModel, TorchSequenceSignalModel, XGBoostSignalModel
from modules.feature_engineering import build_supervised_frame, select_feature_columns
from modules.signals import SignalEngine
from modules.training_utils import split_train_calibration_frame, target_frame


class WalkForwardEngine:
    def __init__(self, settings: Settings, signal_engine: SignalEngine):
        self.settings = settings
        self.signal_engine = signal_engine
        self.backtest_engine = BacktestEngine(settings, signal_engine)

    def _metrics(self, truth: pd.Series, probabilities: pd.Series) -> dict[str, Any]:
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        except ImportError as exc:
            raise RuntimeError("scikit-learn is not installed") from exc

        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(truth, predictions)),
            "precision": float(precision_score(truth, predictions, zero_division=0)),
            "recall": float(recall_score(truth, predictions, zero_division=0)),
        }
        metrics["roc_auc"] = float(roc_auc_score(truth, probabilities)) if truth.nunique() > 1 else None
        return metrics

    def report_path_for(self, symbol: str, now: datetime | None = None) -> Path:
        return self.settings.walk_forward_report_path_for(symbol, now=now)

    def write_report(self, symbol: str, report: dict[str, Any], now: datetime | None = None) -> Path:
        path = self.report_path_for(symbol, now=now)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    def _fold_total_return_pct(self, fold: dict[str, Any]) -> float:
        value = fold.get("backtest", {}).get("Total Return [%]")
        return float(value) if value is not None else 0.0

    def _fold_objective(self, fold: dict[str, Any]) -> float:
        value = fold.get("threshold_tuning", {}).get("objective")
        return float(value) if value is not None else 0.0

    def _regime_snapshot(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {
                "label": "unknown",
                "sideway_share": 0.0,
                "strong_trend_share": 0.0,
                "atr_pct_median": 0.0,
                "ema_trend_strength_median": 0.0,
            }
        trend_strength = frame.get("ema_trend_strength", pd.Series(0.0, index=frame.index)).fillna(0.0).abs()
        atr_pct = frame.get("atr_pct", pd.Series(0.0, index=frame.index)).fillna(0.0)
        sideway_share = float((trend_strength <= 0.35).mean())
        strong_trend_share = float((trend_strength >= 1.25).mean())
        if strong_trend_share >= 0.35:
            label = "strong_trend"
        elif sideway_share >= 0.50:
            label = "sideway"
        else:
            label = "mixed"
        return {
            "label": label,
            "sideway_share": sideway_share,
            "strong_trend_share": strong_trend_share,
            "atr_pct_median": float(atr_pct.median()),
            "ema_trend_strength_median": float(trend_strength.median()),
        }

    def recommend_profile(self, symbol: str, report: dict[str, Any]) -> dict[str, Any]:
        folds = list(report.get("folds") or [])
        min_positive_return_folds = max(1, int(self.settings.model.walk_forward_min_positive_return_folds))
        min_positive_return_pct = float(self.settings.model.walk_forward_min_positive_return_pct)
        positive_return_folds = [
            fold for fold in folds if self._fold_total_return_pct(fold) > min_positive_return_pct
        ]
        positive_objective_folds = [fold for fold in folds if self._fold_objective(fold) > 0]
        accepted = len(positive_return_folds) >= min_positive_return_folds
        profile: dict[str, Any] = {
            "symbol": symbol.upper(),
            "profile_accepted": accepted,
            "profile_source": "walk_forward_consensus" if accepted else "walk_forward_rejected",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "walk_forward_fold_count": len(folds),
            "walk_forward_positive_return_folds": len(positive_return_folds),
            "walk_forward_positive_objective_folds": len(positive_objective_folds),
            "walk_forward_min_positive_return_folds": min_positive_return_folds,
            "walk_forward_positive_return_threshold_pct": min_positive_return_pct,
            "walk_forward_mean_accuracy": report.get("summary", {}).get("mean_accuracy"),
            "walk_forward_mean_roc_auc": report.get("summary", {}).get("mean_roc_auc"),
        }
        regime_counts = report.get("summary", {}).get("regime_counts")
        if isinstance(regime_counts, dict):
            covered = {str(name) for name, count in regime_counts.items() if int(count or 0) > 0}
            missing_required_regimes = sorted({"sideway", "strong_trend"} - covered)
            profile["walk_forward_regime_counts"] = regime_counts
            profile["walk_forward_required_regimes"] = ["sideway", "strong_trend"]
            profile["walk_forward_missing_regimes"] = missing_required_regimes
            profile["walk_forward_regime_coverage_ok"] = not missing_required_regimes
            if missing_required_regimes:
                profile["profile_accepted"] = False
                profile["profile_source"] = "walk_forward_rejected"
                profile["rejection_reason"] = (
                    "Missing required market regimes: " + ",".join(missing_required_regimes)
                )
                return profile
        if not accepted:
            profile["rejection_reason"] = (
                f"Only {len(positive_return_folds)} positive-return folds "
                f"(required {min_positive_return_folds})"
            )
            return profile

        tuning_keys = (
            "long_threshold",
            "short_threshold",
            "risk_per_trade",
            "atr_stop_loss_mult",
            "atr_take_profit_mult",
            "lightgbm_probability_weight",
            "sequence_probability_weight",
            "transformer_probability_weight",
        )
        for key in tuning_keys:
            values = [
                float(fold["threshold_tuning"][key])
                for fold in positive_return_folds
                if key in fold.get("threshold_tuning", {})
            ]
            if values:
                profile[key] = float(median(values))
        return_values = [self._fold_total_return_pct(fold) for fold in positive_return_folds]
        objective_values = [self._fold_objective(fold) for fold in positive_return_folds]
        profile["walk_forward_selected_folds"] = [int(fold.get("fold", 0)) for fold in positive_return_folds]
        profile["walk_forward_selected_return_median"] = float(median(return_values))
        profile["walk_forward_selected_objective_median"] = float(median(objective_values))
        return profile

    def run(self, symbol: str, market_frame: pd.DataFrame) -> dict[str, Any]:
        training_frame = build_supervised_frame(
            market_frame,
            horizon=self.settings.model.positive_return_horizon,
            edge_bps=self.settings.model.positive_return_bps,
            breakout_pct=self.settings.model.volatility_breakout_pct,
            symbol=symbol,
        )
        total_rows = len(training_frame)
        if total_rows < self.settings.model.training_min_rows:
            raise ValueError("Not enough rows for walk-forward analysis")

        splits = max(2, self.settings.model.walk_forward_splits)
        initial_train = max(self.settings.model.training_min_rows, total_rows // 2)
        remaining = total_rows - initial_train
        test_size = max(100, remaining // splits)
        folds: list[dict[str, Any]] = []

        for fold in range(splits):
            train_end = initial_train + fold * test_size
            test_end = min(train_end + test_size, total_rows)
            if test_end - train_end < 50:
                break

            train_full = training_frame.iloc[:train_end].copy()
            test = training_frame.iloc[train_end:test_end].copy()
            calibration_share = self.settings.model.calibration_ratio / max(
                self.settings.model.training_ratio + self.settings.model.calibration_ratio,
                1e-9,
            )
            train, calibration = split_train_calibration_frame(train_full, calibration_share=calibration_share)

            long_model = XGBoostSignalModel(artifact_path=None, target_side="long")
            short_model = XGBoostSignalModel(artifact_path=None, target_side="short")
            lgbm_long_model = (
                LightGBMSignalModel(artifact_path=None, target_side="long")
                if self.settings.model.lightgbm_enabled
                else None
            )
            lgbm_short_model = (
                LightGBMSignalModel(artifact_path=None, target_side="short")
                if self.settings.model.lightgbm_enabled
                else None
            )
            sequence_long_model = (
                TorchSequenceSignalModel(
                    artifact_path=None,
                    target_side="long",
                    sequence_length=self.settings.model.sequence_length,
                    hidden_size=self.settings.model.sequence_hidden_size,
                    num_layers=self.settings.model.sequence_num_layers,
                    dropout=self.settings.model.sequence_dropout,
                    learning_rate=self.settings.model.sequence_learning_rate,
                    epochs=self.settings.model.sequence_epochs,
                    batch_size=self.settings.model.sequence_batch_size,
                    device=self.settings.model.sequence_device,
                )
                if self.settings.model.sequence_enabled
                else None
            )
            sequence_short_model = (
                TorchSequenceSignalModel(
                    artifact_path=None,
                    target_side="short",
                    sequence_length=self.settings.model.sequence_length,
                    hidden_size=self.settings.model.sequence_hidden_size,
                    num_layers=self.settings.model.sequence_num_layers,
                    dropout=self.settings.model.sequence_dropout,
                    learning_rate=self.settings.model.sequence_learning_rate,
                    epochs=self.settings.model.sequence_epochs,
                    batch_size=self.settings.model.sequence_batch_size,
                    device=self.settings.model.sequence_device,
                )
                if self.settings.model.sequence_enabled
                else None
            )
            transformer_long_model = (
                TorchSequenceSignalModel(
                    artifact_path=None,
                    target_side="long",
                    sequence_length=self.settings.model.sequence_length,
                    hidden_size=self.settings.model.sequence_hidden_size,
                    num_layers=self.settings.model.sequence_num_layers,
                    dropout=self.settings.model.sequence_dropout,
                    learning_rate=self.settings.model.sequence_learning_rate,
                    epochs=self.settings.model.sequence_epochs,
                    batch_size=self.settings.model.sequence_batch_size,
                    device=self.settings.model.sequence_device,
                    architecture="transformer",
                    transformer_heads=self.settings.model.transformer_heads,
                    transformer_ffn_size=self.settings.model.transformer_ffn_size,
                )
                if self.settings.model.transformer_enabled
                else None
            )
            transformer_short_model = (
                TorchSequenceSignalModel(
                    artifact_path=None,
                    target_side="short",
                    sequence_length=self.settings.model.sequence_length,
                    hidden_size=self.settings.model.sequence_hidden_size,
                    num_layers=self.settings.model.sequence_num_layers,
                    dropout=self.settings.model.sequence_dropout,
                    learning_rate=self.settings.model.sequence_learning_rate,
                    epochs=self.settings.model.sequence_epochs,
                    batch_size=self.settings.model.sequence_batch_size,
                    device=self.settings.model.sequence_device,
                    architecture="transformer",
                    transformer_heads=self.settings.model.transformer_heads,
                    transformer_ffn_size=self.settings.model.transformer_ffn_size,
                )
                if self.settings.model.transformer_enabled
                else None
            )
            long_train = target_frame(train, "target_long")
            short_train = target_frame(train, "target_short")
            long_calibration = target_frame(calibration, "target_long")
            short_calibration = target_frame(calibration, "target_short")

            if self.settings.model.feature_selection_enabled:
                long_features = select_feature_columns(
                    long_train,
                    target_column="target",
                    min_features=self.settings.model.feature_selection_min_features,
                    max_features=self.settings.model.feature_selection_max_features,
                    forced_features=list(self.settings.model.feature_selection_forced_features),
                    correlation_threshold=self.settings.model.feature_selection_correlation_threshold,
                    min_mutual_info=self.settings.model.feature_selection_min_mutual_info,
                )
                short_features = select_feature_columns(
                    short_train,
                    target_column="target",
                    min_features=self.settings.model.feature_selection_min_features,
                    max_features=self.settings.model.feature_selection_max_features,
                    forced_features=list(self.settings.model.feature_selection_forced_features),
                    correlation_threshold=self.settings.model.feature_selection_correlation_threshold,
                    min_mutual_info=self.settings.model.feature_selection_min_mutual_info,
                )
            else:
                long_features = list(long_model.feature_columns)
                short_features = list(short_model.feature_columns)

            for model, selected_features in (
                (long_model, long_features),
                (short_model, short_features),
                (lgbm_long_model, long_features),
                (lgbm_short_model, short_features),
                (sequence_long_model, long_features),
                (sequence_short_model, short_features),
                (transformer_long_model, long_features),
                (transformer_short_model, short_features),
            ):
                if model is not None:
                    model.feature_columns = list(selected_features)

            long_model.fit(long_train)
            short_model.fit(short_train)
            if lgbm_long_model is not None and lgbm_short_model is not None:
                lgbm_long_model.fit(long_train)
                lgbm_short_model.fit(short_train)
            if sequence_long_model is not None and sequence_short_model is not None:
                sequence_long_model.fit(long_train)
                sequence_short_model.fit(short_train)
            if transformer_long_model is not None and transformer_short_model is not None:
                transformer_long_model.fit(long_train)
                transformer_short_model.fit(short_train)
            long_calibration_summary = long_model.fit_calibrator(
                long_calibration,
                method=self.settings.model.calibration_method,
            )
            short_calibration_summary = short_model.fit_calibrator(
                short_calibration,
                method=self.settings.model.calibration_method,
            )
            tuned = self.backtest_engine.optimize_parameters(
                symbol,
                calibration,
                long_model.predict_proba_frame(calibration),
                short_model.predict_proba_frame(calibration),
                lightgbm_long_probabilities=(
                    lgbm_long_model.predict_proba_frame(calibration)
                    if lgbm_long_model is not None and lgbm_short_model is not None
                    else None
                ),
                lightgbm_short_probabilities=(
                    lgbm_short_model.predict_proba_frame(calibration)
                    if lgbm_long_model is not None and lgbm_short_model is not None
                    else None
                ),
                sequence_long_probabilities=(
                    sequence_long_model.predict_proba_frame(calibration)
                    if sequence_long_model is not None and sequence_short_model is not None
                    else None
                ),
                sequence_short_probabilities=(
                    sequence_short_model.predict_proba_frame(calibration)
                    if sequence_long_model is not None and sequence_short_model is not None
                    else None
                ),
                transformer_long_probabilities=(
                    transformer_long_model.predict_proba_frame(calibration)
                    if transformer_long_model is not None and transformer_short_model is not None
                    else None
                ),
                transformer_short_probabilities=(
                    transformer_short_model.predict_proba_frame(calibration)
                    if transformer_long_model is not None and transformer_short_model is not None
                    else None
                ),
            )
            long_model.set_threshold(tuned["long_threshold"])
            short_model.set_threshold(tuned["short_threshold"])

            long_probabilities = long_model.predict_proba_frame(test)
            short_probabilities = short_model.predict_proba_frame(test)
            if (
                (lgbm_long_model is not None and lgbm_short_model is not None)
                or (sequence_long_model is not None and sequence_short_model is not None)
                or (transformer_long_model is not None and transformer_short_model is not None)
            ):
                long_probabilities, short_probabilities = self.backtest_engine._blend_probability_streams(
                    long_probabilities,
                    short_probabilities,
                    lightgbm_long_probabilities=(
                        lgbm_long_model.predict_proba_frame(test)
                        if lgbm_long_model is not None and lgbm_short_model is not None
                        else None
                    ),
                    lightgbm_short_probabilities=(
                        lgbm_short_model.predict_proba_frame(test)
                        if lgbm_long_model is not None and lgbm_short_model is not None
                        else None
                    ),
                    lightgbm_weight=tuned.get(
                        "lightgbm_probability_weight",
                        self.settings.lightgbm_probability_weight_for(symbol),
                    ),
                    sequence_long_probabilities=(
                        sequence_long_model.predict_proba_frame(test)
                        if sequence_long_model is not None and sequence_short_model is not None
                        else None
                    ),
                    sequence_short_probabilities=(
                        sequence_short_model.predict_proba_frame(test)
                        if sequence_long_model is not None and sequence_short_model is not None
                        else None
                    ),
                    sequence_weight=tuned.get(
                        "sequence_probability_weight",
                        self.settings.sequence_probability_weight_for(symbol),
                    ),
                    transformer_long_probabilities=(
                        transformer_long_model.predict_proba_frame(test)
                        if transformer_long_model is not None and transformer_short_model is not None
                        else None
                    ),
                    transformer_short_probabilities=(
                        transformer_short_model.predict_proba_frame(test)
                        if transformer_long_model is not None and transformer_short_model is not None
                        else None
                    ),
                    transformer_weight=tuned.get(
                        "transformer_probability_weight",
                        self.settings.transformer_probability_weight_for(symbol),
                    ),
                )
            fold_metrics = {
                "long": long_model.evaluate(
                    target_frame(test, "target_long"),
                    threshold=long_model.effective_threshold(self.settings.threshold_for(symbol, "long")),
                ),
                "short": short_model.evaluate(
                    target_frame(test, "target_short"),
                    threshold=short_model.effective_threshold(self.settings.threshold_for(symbol, "short")),
                ),
            }
            backtest = self.backtest_engine.run_from_probabilities(
                symbol,
                test,
                long_probabilities,
                short_probabilities,
                long_threshold=long_model.effective_threshold(self.settings.threshold_for(symbol, "long")),
                short_threshold=short_model.effective_threshold(self.settings.threshold_for(symbol, "short")),
                risk_per_trade=tuned.get("risk_per_trade"),
                atr_stop_loss_mult=tuned.get("atr_stop_loss_mult"),
                atr_take_profit_mult=tuned.get("atr_take_profit_mult"),
            )
            folds.append(
                {
                    "fold": fold + 1,
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                    "test_rows": int(len(test)),
                    "market_regime": self._regime_snapshot(test),
                    "train_summary": {
                        "long": {
                            "rows_train": int(len(long_train)),
                            "positive_rate_train": float(long_train["target"].mean()),
                            **long_calibration_summary,
                        },
                        "short": {
                            "rows_train": int(len(short_train)),
                            "positive_rate_train": float(short_train["target"].mean()),
                            **short_calibration_summary,
                        },
                    },
                    "threshold_tuning": {
                        "long_threshold": tuned["long_threshold"],
                        "short_threshold": tuned["short_threshold"],
                        "risk_per_trade": tuned["risk_per_trade"],
                        "atr_stop_loss_mult": tuned["atr_stop_loss_mult"],
                        "atr_take_profit_mult": tuned["atr_take_profit_mult"],
                        "lightgbm_probability_weight": tuned.get("lightgbm_probability_weight", 0.0),
                        "sequence_probability_weight": tuned.get("sequence_probability_weight", 0.0),
                        "transformer_probability_weight": tuned.get("transformer_probability_weight", 0.0),
                        "objective": tuned["objective"],
                    },
                    "test_metrics": fold_metrics,
                    "backtest": {
                        "Total Return [%]": backtest.get("Total Return [%]"),
                        "Win Rate [%]": backtest.get("Win Rate [%]"),
                        "Sharpe Ratio": backtest.get("Sharpe Ratio"),
                        "Max Drawdown [%]": backtest.get("Max Drawdown [%]"),
                    },
                }
            )

        auc_values = []
        accuracy_values = []
        regime_counts: dict[str, int] = {}
        for fold in folds:
            regime_name = str(fold.get("market_regime", {}).get("label", "unknown"))
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
            for side in ("long", "short"):
                metrics = fold["test_metrics"][side]
                accuracy_values.append(metrics["accuracy"])
                if metrics["roc_auc"] is not None:
                    auc_values.append(metrics["roc_auc"])
        return {
            "folds": folds,
            "summary": {
                "fold_count": len(folds),
                "mean_accuracy": float(np.mean(accuracy_values)) if accuracy_values else None,
                "mean_roc_auc": float(np.mean(auc_values)) if auc_values else None,
                "regime_counts": regime_counts,
                "regime_coverage_ok": (
                    regime_counts.get("sideway", 0) > 0 and regime_counts.get("strong_trend", 0) > 0
                ),
            },
        }

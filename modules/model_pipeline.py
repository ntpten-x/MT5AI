from __future__ import annotations

from typing import Any

import pandas as pd

from config import Settings
from modules.backtest import BacktestEngine
from modules.brain_ai import LightGBMSignalModel, TorchSequenceSignalModel, XGBoostSignalModel
from modules.feature_engineering import select_feature_columns
from modules.training_utils import split_time_series_frame, target_frame


def _benchmark_winner(xgb_auc: Any, lgbm_auc: Any) -> str:
    xgb_value = float(xgb_auc) if xgb_auc is not None else None
    lgbm_value = float(lgbm_auc) if lgbm_auc is not None else None
    if xgb_value is None and lgbm_value is None:
        return "insufficient_data"
    if xgb_value is None:
        return "lightgbm"
    if lgbm_value is None:
        return "xgboost"
    if abs(xgb_value - lgbm_value) < 1e-9:
        return "tie"
    return "xgboost" if xgb_value > lgbm_value else "lightgbm"


def train_dual_models(
    symbol: str,
    supervised_frame: pd.DataFrame,
    settings: Settings,
    backtest_engine: BacktestEngine,
    long_model: XGBoostSignalModel,
    short_model: XGBoostSignalModel,
    lgbm_long_model: LightGBMSignalModel | None = None,
    lgbm_short_model: LightGBMSignalModel | None = None,
    sequence_long_model: TorchSequenceSignalModel | None = None,
    sequence_short_model: TorchSequenceSignalModel | None = None,
    transformer_long_model: TorchSequenceSignalModel | None = None,
    transformer_short_model: TorchSequenceSignalModel | None = None,
) -> dict[str, Any]:
    train, calibration, test = split_time_series_frame(
        supervised_frame,
        train_ratio=settings.model.training_ratio,
        calibration_ratio=settings.model.calibration_ratio,
    )

    long_train = target_frame(train, "target_long")
    short_train = target_frame(train, "target_short")
    long_calibration = target_frame(calibration, "target_long")
    short_calibration = target_frame(calibration, "target_short")
    long_test = target_frame(test, "target_long")
    short_test = target_frame(test, "target_short")

    if settings.model.feature_selection_enabled:
        long_features = select_feature_columns(
            long_train,
            target_column="target",
            min_features=settings.model.feature_selection_min_features,
            max_features=settings.model.feature_selection_max_features,
            forced_features=list(settings.model.feature_selection_forced_features),
            correlation_threshold=settings.model.feature_selection_correlation_threshold,
            min_mutual_info=settings.model.feature_selection_min_mutual_info,
        )
        short_features = select_feature_columns(
            short_train,
            target_column="target",
            min_features=settings.model.feature_selection_min_features,
            max_features=settings.model.feature_selection_max_features,
            forced_features=list(settings.model.feature_selection_forced_features),
            correlation_threshold=settings.model.feature_selection_correlation_threshold,
            min_mutual_info=settings.model.feature_selection_min_mutual_info,
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
    lightgbm_enabled = lgbm_long_model is not None and lgbm_short_model is not None
    if lightgbm_enabled:
        lgbm_long_model.fit(long_train)
        lgbm_short_model.fit(short_train)
    sequence_enabled = sequence_long_model is not None and sequence_short_model is not None
    transformer_enabled = transformer_long_model is not None and transformer_short_model is not None
    if sequence_enabled:
        sequence_long_model.fit(long_train)
        sequence_short_model.fit(short_train)
    if transformer_enabled:
        transformer_long_model.fit(long_train)
        transformer_short_model.fit(short_train)
    long_calibration_summary = long_model.fit_calibrator(
        long_calibration,
        method=settings.model.calibration_method,
    )
    short_calibration_summary = short_model.fit_calibrator(
        short_calibration,
        method=settings.model.calibration_method,
    )

    long_calibration_probabilities = long_model.predict_proba_frame(calibration)
    short_calibration_probabilities = short_model.predict_proba_frame(calibration)
    lightgbm_long_calibration_probabilities = (
        lgbm_long_model.predict_proba_frame(calibration) if lightgbm_enabled else None
    )
    lightgbm_short_calibration_probabilities = (
        lgbm_short_model.predict_proba_frame(calibration) if lightgbm_enabled else None
    )
    sequence_long_calibration_probabilities = (
        sequence_long_model.predict_proba_frame(calibration) if sequence_enabled else None
    )
    sequence_short_calibration_probabilities = (
        sequence_short_model.predict_proba_frame(calibration) if sequence_enabled else None
    )
    transformer_long_calibration_probabilities = (
        transformer_long_model.predict_proba_frame(calibration) if transformer_enabled else None
    )
    transformer_short_calibration_probabilities = (
        transformer_short_model.predict_proba_frame(calibration) if transformer_enabled else None
    )
    tuned = backtest_engine.optimize_parameters(
        symbol,
        calibration,
        long_calibration_probabilities,
        short_calibration_probabilities,
        lightgbm_long_probabilities=lightgbm_long_calibration_probabilities,
        lightgbm_short_probabilities=lightgbm_short_calibration_probabilities,
        sequence_long_probabilities=sequence_long_calibration_probabilities,
        sequence_short_probabilities=sequence_short_calibration_probabilities,
        transformer_long_probabilities=transformer_long_calibration_probabilities,
        transformer_short_probabilities=transformer_short_calibration_probabilities,
    )
    long_model.set_threshold(tuned["long_threshold"])
    short_model.set_threshold(tuned["short_threshold"])

    long_test_metrics = long_model.evaluate(
        long_test,
        threshold=long_model.effective_threshold(settings.threshold_for(symbol, "long")),
    )
    short_test_metrics = short_model.evaluate(
        short_test,
        threshold=short_model.effective_threshold(settings.threshold_for(symbol, "short")),
    )
    lightgbm_long_test_metrics = (
        lgbm_long_model.evaluate(
            long_test,
            threshold=long_model.effective_threshold(settings.threshold_for(symbol, "long")),
        )
        if lightgbm_enabled
        else {}
    )
    lightgbm_short_test_metrics = (
        lgbm_short_model.evaluate(
            short_test,
            threshold=short_model.effective_threshold(settings.threshold_for(symbol, "short")),
        )
        if lightgbm_enabled
        else {}
    )
    sequence_long_test_metrics = (
        sequence_long_model.evaluate(
            long_test,
            threshold=long_model.effective_threshold(settings.threshold_for(symbol, "long")),
        )
        if sequence_enabled
        else {}
    )
    sequence_short_test_metrics = (
        sequence_short_model.evaluate(
            short_test,
            threshold=short_model.effective_threshold(settings.threshold_for(symbol, "short")),
        )
        if sequence_enabled
        else {}
    )
    transformer_long_test_metrics = (
        transformer_long_model.evaluate(
            long_test,
            threshold=long_model.effective_threshold(settings.threshold_for(symbol, "long")),
        )
        if transformer_enabled
        else {}
    )
    transformer_short_test_metrics = (
        transformer_short_model.evaluate(
            short_test,
            threshold=short_model.effective_threshold(settings.threshold_for(symbol, "short")),
        )
        if transformer_enabled
        else {}
    )
    sequence_long_test_probabilities = (
        sequence_long_model.predict_proba_frame(test) if sequence_enabled else None
    )
    sequence_short_test_probabilities = (
        sequence_short_model.predict_proba_frame(test) if sequence_enabled else None
    )
    transformer_long_test_probabilities = (
        transformer_long_model.predict_proba_frame(test) if transformer_enabled else None
    )
    transformer_short_test_probabilities = (
        transformer_short_model.predict_proba_frame(test) if transformer_enabled else None
    )
    blended_long_test_probabilities, blended_short_test_probabilities = backtest_engine._blend_probability_streams(
        long_model.predict_proba_frame(test),
        short_model.predict_proba_frame(test),
        lightgbm_long_probabilities=(lgbm_long_model.predict_proba_frame(test) if lightgbm_enabled else None),
        lightgbm_short_probabilities=(lgbm_short_model.predict_proba_frame(test) if lightgbm_enabled else None),
        lightgbm_weight=tuned.get("lightgbm_probability_weight", settings.lightgbm_probability_weight_for(symbol)),
        sequence_long_probabilities=sequence_long_test_probabilities,
        sequence_short_probabilities=sequence_short_test_probabilities,
        sequence_weight=tuned.get("sequence_probability_weight", settings.sequence_probability_weight_for(symbol)),
        transformer_long_probabilities=transformer_long_test_probabilities,
        transformer_short_probabilities=transformer_short_test_probabilities,
        transformer_weight=tuned.get(
            "transformer_probability_weight",
            settings.transformer_probability_weight_for(symbol),
        ),
    )
    backtest = backtest_engine.run_from_probabilities(
        symbol,
        test,
        blended_long_test_probabilities,
        blended_short_test_probabilities,
        long_threshold=long_model.effective_threshold(settings.threshold_for(symbol, "long")),
        short_threshold=short_model.effective_threshold(settings.threshold_for(symbol, "short")),
        risk_per_trade=tuned.get("risk_per_trade"),
        atr_stop_loss_mult=tuned.get("atr_stop_loss_mult"),
        atr_take_profit_mult=tuned.get("atr_take_profit_mult"),
    )

    long_model.save()
    short_model.save()
    if lightgbm_enabled:
        lgbm_long_model.save()
        lgbm_short_model.save()
    if sequence_enabled:
        sequence_long_model.save()
        sequence_short_model.save()
    if transformer_enabled:
        transformer_long_model.save()
        transformer_short_model.save()

    return {
        "splits": {
            "rows_train": int(len(train)),
            "rows_calibration": int(len(calibration)),
            "rows_test": int(len(test)),
        },
        "threshold_tuning": {
            "long_threshold": float(tuned["long_threshold"]),
            "short_threshold": float(tuned["short_threshold"]),
            "risk_per_trade": float(tuned["risk_per_trade"]),
            "atr_stop_loss_mult": float(tuned["atr_stop_loss_mult"]),
            "atr_take_profit_mult": float(tuned["atr_take_profit_mult"]),
            "lightgbm_probability_weight": float(tuned.get("lightgbm_probability_weight", 0.0)),
            "sequence_probability_weight": float(tuned.get("sequence_probability_weight", 0.0)),
            "transformer_probability_weight": float(tuned.get("transformer_probability_weight", 0.0)),
            "objective": float(tuned["objective"]),
            "calibration_backtest": {
                "Total Return [%]": tuned["stats"].get("Total Return [%]"),
                "Sharpe Ratio": tuned["stats"].get("Sharpe Ratio"),
                "Max Drawdown [%]": tuned["stats"].get("Max Drawdown [%]"),
                "Total Trades": tuned["stats"].get("Total Trades"),
            },
        },
        "long": {
            "rows_train": int(len(long_train)),
            "positive_rate_train": float(long_train["target"].mean()),
            **long_calibration_summary,
            **long_test_metrics,
            "artifact_path": str(long_model.artifact_path) if long_model.artifact_path else None,
        },
        "short": {
            "rows_train": int(len(short_train)),
            "positive_rate_train": float(short_train["target"].mean()),
            **short_calibration_summary,
            **short_test_metrics,
            "artifact_path": str(short_model.artifact_path) if short_model.artifact_path else None,
        },
        "lightgbm_long": {
            "enabled": lightgbm_enabled,
            **lightgbm_long_test_metrics,
            "artifact_path": str(lgbm_long_model.artifact_path) if lightgbm_enabled and lgbm_long_model.artifact_path else None,
        },
        "lightgbm_short": {
            "enabled": lightgbm_enabled,
            **lightgbm_short_test_metrics,
            "artifact_path": str(lgbm_short_model.artifact_path) if lightgbm_enabled and lgbm_short_model.artifact_path else None,
        },
        "sequence_long": {
            "enabled": sequence_enabled,
            **sequence_long_test_metrics,
            "artifact_path": str(sequence_long_model.artifact_path) if sequence_enabled and sequence_long_model.artifact_path else None,
        },
        "sequence_short": {
            "enabled": sequence_enabled,
            **sequence_short_test_metrics,
            "artifact_path": str(sequence_short_model.artifact_path) if sequence_enabled and sequence_short_model.artifact_path else None,
        },
        "transformer_long": {
            "enabled": transformer_enabled,
            **transformer_long_test_metrics,
            "artifact_path": str(transformer_long_model.artifact_path)
            if transformer_enabled and transformer_long_model.artifact_path
            else None,
        },
        "transformer_short": {
            "enabled": transformer_enabled,
            **transformer_short_test_metrics,
            "artifact_path": str(transformer_short_model.artifact_path)
            if transformer_enabled and transformer_short_model.artifact_path
            else None,
        },
        "feature_selection": {
            "enabled": settings.model.feature_selection_enabled,
            "long_features": long_features,
            "short_features": short_features,
        },
        "benchmark": {
            "tabular_primary": "xgboost",
            "xgboost_long_roc_auc": long_test_metrics.get("roc_auc"),
            "xgboost_short_roc_auc": short_test_metrics.get("roc_auc"),
            "lightgbm_long_roc_auc": lightgbm_long_test_metrics.get("roc_auc"),
            "lightgbm_short_roc_auc": lightgbm_short_test_metrics.get("roc_auc"),
            "long_winner": _benchmark_winner(long_test_metrics.get("roc_auc"), lightgbm_long_test_metrics.get("roc_auc")),
            "short_winner": _benchmark_winner(
                short_test_metrics.get("roc_auc"),
                lightgbm_short_test_metrics.get("roc_auc"),
            ),
        },
        "test_backtest": {
            "Total Return [%]": backtest.get("Total Return [%]"),
            "Sharpe Ratio": backtest.get("Sharpe Ratio"),
            "Max Drawdown [%]": backtest.get("Max Drawdown [%]"),
            "Total Trades": backtest.get("Total Trades"),
        },
    }

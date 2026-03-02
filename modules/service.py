from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import Settings
from modules.backtest import BacktestEngine
from modules.brain_ai import ChronosForecaster, LightGBMSignalModel, TorchSequenceSignalModel, XGBoostSignalModel
from modules.db import Database
from modules.execution import ExecutionEngine
from modules.feature_engineering import FEATURE_COLUMNS, build_feature_frame, build_supervised_frame
from modules.logging_setup import configure_logging
from modules.model_pipeline import train_dual_models
from modules.mt5_bridge import MT5Bridge
from modules.news_filter import NewsFilter
from modules.notifications import TelegramNotifier
from modules.risk_control import RiskManager
from modules.runtime_lock import SingleInstanceLock
from modules.signals import SignalEngine, heuristic_probabilities
from modules.spread_profile import SpreadProfileManager
from modules.timeframes import scheduler_trigger_args
from modules.walk_forward import WalkForwardEngine


class TradingBotService:
    def __init__(self, settings: Settings):
        self.settings = settings
        configure_logging(settings)
        self.database = Database(settings.database_path)
        self.database.initialize()
        self.bridge = MT5Bridge(settings)
        self.notifier = TelegramNotifier(
            enabled=settings.telegram.enabled,
            bot_token=settings.telegram.bot_token,
            chat_id=settings.telegram.chat_id,
        )
        self.risk_manager = RiskManager(settings, self.database)
        self.signal_engine = SignalEngine(settings)
        self.news_filter = NewsFilter(settings)
        self.spread_profile_manager = SpreadProfileManager(settings)
        self.execution_engine = ExecutionEngine(
            settings=settings,
            bridge=self.bridge,
            database=self.database,
            risk_manager=self.risk_manager,
            notifier=self.notifier,
        )
        self.xgb_models: dict[tuple[str, str], XGBoostSignalModel] = {}
        self.lgbm_models: dict[tuple[str, str], LightGBMSignalModel] = {}
        self.sequence_models: dict[tuple[str, str], TorchSequenceSignalModel] = {}
        self.transformer_models: dict[tuple[str, str], TorchSequenceSignalModel] = {}
        self.chronos = ChronosForecaster(
            enabled=settings.model.chronos_enabled,
            model_id=settings.model.chronos_model_id,
            device=settings.model.chronos_device,
        )
        self.backtest_engine = BacktestEngine(settings, self.signal_engine)
        self.walk_forward_engine = WalkForwardEngine(settings, self.signal_engine)
        self._last_heartbeat_ok: bool | None = None
        self._run_lock = SingleInstanceLock(settings.run_lock_path)

    def init_database(self) -> None:
        self.database.initialize()

    def reset_kill_switch(self) -> dict[str, Any]:
        previous = self.risk_manager.kill_switch_status()
        self.risk_manager.reset_kill_switch()
        payload = {
            "status": "reset",
            "kill_switch_was_active": previous is not None,
            "previous": previous,
        }
        self.database.record_event("INFO", "kill_switch_reset", "Kill switch reset via CLI", payload)
        self.notifier.send_warning("Kill switch reset via CLI")
        return payload

    def kill_switch_status(self) -> dict[str, Any]:
        payload = self.risk_manager.kill_switch_status()
        return {
            "active": payload is not None,
            "kill_switch": payload,
            "path": str(self.settings.kill_switch_path),
        }

    def connect(self) -> bool:
        return self.bridge.connect()

    def shutdown(self) -> None:
        self._run_lock.release()
        self.bridge.shutdown()

    def _symbols(self, symbol: str | None = None) -> list[str]:
        return [symbol] if symbol else list(self.settings.trading.symbols)

    def _watchdog_or_raise(self) -> None:
        heartbeat_state = self.bridge.heartbeat()
        if not heartbeat_state.ok:
            reason = f"Watchdog halted: MT5 connection unavailable ({heartbeat_state.message})"
            self.database.record_event("ERROR", "watchdog_mt5", reason)
            self.notifier.send_critical(reason)
            raise RuntimeError(reason)
        if not self.news_filter.internet_available():
            reason = "Watchdog halted: internet/news endpoint unavailable"
            self.database.record_event("ERROR", "watchdog_network", reason)
            self.notifier.send_critical(reason)
            raise RuntimeError(reason)

    def _reference_frames(self) -> dict[str, Any]:
        refs: dict[str, Any] = {}
        count = self.settings.trading.history_bars
        for ref_symbol in {"BTCUSD", "ETHUSD", self.settings.trading.dxy_symbol.upper()}:
            if not ref_symbol:
                continue
            try:
                frame = self.bridge.fetch_rates(ref_symbol, self.settings.trading.timeframe, count)
                if not frame.empty:
                    refs[ref_symbol] = frame
            except Exception:
                continue
        return refs

    def _heuristic_probabilities(self, feature_frame) -> tuple[float, float]:
        return heuristic_probabilities(feature_frame.iloc[-1])

    def _xgb_model(self, symbol: str, target_side: str) -> XGBoostSignalModel:
        key = (symbol, target_side)
        if key not in self.xgb_models:
            self.xgb_models[key] = XGBoostSignalModel(
                self.settings.xgb_artifact_path_for(symbol, target_side=target_side),
                target_side=target_side,
            )
        return self.xgb_models[key]

    def _xgb_probabilities(self, symbol: str, feature_frame) -> tuple[float, float]:
        long_model = self._xgb_model(symbol, "long")
        short_model = self._xgb_model(symbol, "short")
        if long_model.is_ready() and short_model.is_ready():
            try:
                return long_model.predict_latest(feature_frame), short_model.predict_latest(feature_frame)
            except Exception as exc:
                logger.warning("XGBoost inference failed, falling back to heuristic: {}", exc)
        return self._heuristic_probabilities(feature_frame)

    def _lgbm_model(self, symbol: str, target_side: str) -> LightGBMSignalModel:
        key = (symbol, target_side)
        if key not in self.lgbm_models:
            self.lgbm_models[key] = LightGBMSignalModel(
                self.settings.lgbm_artifact_path_for(symbol, target_side=target_side),
                target_side=target_side,
            )
        return self.lgbm_models[key]

    def _lgbm_probabilities(self, symbol: str, feature_frame) -> tuple[float | None, float | None]:
        if not self.settings.model.lightgbm_enabled:
            return None, None
        long_model = self._lgbm_model(symbol, "long")
        short_model = self._lgbm_model(symbol, "short")
        if long_model.is_ready() and short_model.is_ready():
            try:
                return long_model.predict_latest(feature_frame), short_model.predict_latest(feature_frame)
            except Exception as exc:
                logger.warning("LightGBM inference failed, falling back to XGBoost only: {}", exc)
        return None, None

    def _xgb_thresholds(self, symbol: str) -> tuple[float, float]:
        long_model = self._xgb_model(symbol, "long")
        short_model = self._xgb_model(symbol, "short")
        explicit_long_threshold = self.settings.model.symbol_long_thresholds.get(symbol.upper())
        explicit_short_threshold = self.settings.model.symbol_short_thresholds.get(symbol.upper())
        long_default = self.settings.threshold_for(symbol, "long")
        short_default = self.settings.threshold_for(symbol, "short")
        return (
            float(long_default)
            if explicit_long_threshold is not None
            else long_model.effective_threshold(long_default),
            float(short_default)
            if explicit_short_threshold is not None
            else short_model.effective_threshold(short_default),
        )

    def _sequence_model(self, symbol: str, target_side: str) -> TorchSequenceSignalModel:
        key = (symbol, target_side)
        if key not in self.sequence_models:
            self.sequence_models[key] = TorchSequenceSignalModel(
                artifact_path=self.settings.sequence_artifact_path_for(symbol, target_side=target_side),
                target_side=target_side,
                sequence_length=self.settings.model.sequence_length,
                hidden_size=self.settings.model.sequence_hidden_size,
                num_layers=self.settings.model.sequence_num_layers,
                dropout=self.settings.model.sequence_dropout,
                learning_rate=self.settings.model.sequence_learning_rate,
                epochs=self.settings.model.sequence_epochs,
                batch_size=self.settings.model.sequence_batch_size,
                device=self.settings.model.sequence_device,
            )
        return self.sequence_models[key]

    def _transformer_model(self, symbol: str, target_side: str) -> TorchSequenceSignalModel:
        key = (symbol, target_side)
        if key not in self.transformer_models:
            self.transformer_models[key] = TorchSequenceSignalModel(
                artifact_path=self.settings.transformer_artifact_path_for(symbol, target_side=target_side),
                target_side=target_side,
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
        return self.transformer_models[key]

    def _sequence_probabilities(self, symbol: str, feature_frame) -> tuple[float | None, float | None]:
        if not self.settings.model.sequence_enabled:
            return None, None
        long_model = self._sequence_model(symbol, "long")
        short_model = self._sequence_model(symbol, "short")
        if long_model.is_ready() and short_model.is_ready():
            try:
                return long_model.predict_latest(feature_frame), short_model.predict_latest(feature_frame)
            except Exception as exc:
                logger.warning("Sequence inference failed, falling back to XGBoost only: {}", exc)
        return None, None

    def _transformer_probabilities(self, symbol: str, feature_frame) -> tuple[float | None, float | None]:
        if not self.settings.model.transformer_enabled:
            return None, None
        long_model = self._transformer_model(symbol, "long")
        short_model = self._transformer_model(symbol, "short")
        if long_model.is_ready() and short_model.is_ready():
            try:
                return long_model.predict_latest(feature_frame), short_model.predict_latest(feature_frame)
            except Exception as exc:
                logger.warning("Transformer inference failed, falling back to XGBoost/LSTM only: {}", exc)
        return None, None

    def _blend_live_probabilities(
        self,
        symbol: str,
        xgb_long: float,
        xgb_short: float,
        lgbm_long: float | None,
        lgbm_short: float | None,
        sequence_long: float | None,
        sequence_short: float | None,
        transformer_long: float | None,
        transformer_short: float | None,
        chronos_direction: float | None,
    ) -> tuple[float, float]:
        long_probability = float(np.clip(xgb_long, 0.01, 0.99))
        short_probability = float(np.clip(xgb_short, 0.01, 0.99))
        auxiliary_streams: list[tuple[float | None, float | None, float]] = []
        if lgbm_long is not None and lgbm_short is not None:
            auxiliary_streams.append(
                (
                    lgbm_long,
                    lgbm_short,
                    float(np.clip(self.settings.lightgbm_probability_weight_for(symbol), 0.0, 1.0)),
                )
            )
        if sequence_long is not None and sequence_short is not None:
            auxiliary_streams.append(
                (sequence_long, sequence_short, float(np.clip(self.settings.sequence_probability_weight_for(symbol), 0.0, 1.0)))
            )
        if transformer_long is not None and transformer_short is not None:
            auxiliary_streams.append(
                (
                    transformer_long,
                    transformer_short,
                    float(np.clip(self.settings.transformer_probability_weight_for(symbol), 0.0, 1.0)),
                )
            )
        total_aux_weight = sum(weight for _, _, weight in auxiliary_streams if weight > 0)
        if total_aux_weight > 0:
            normalized_total = min(total_aux_weight, 0.8)
            base_scale = max(0.0, 1.0 - normalized_total)
            blended_long = long_probability * base_scale
            blended_short = short_probability * base_scale
            for aux_long, aux_short, weight in auxiliary_streams:
                if aux_long is None or aux_short is None or weight <= 0:
                    continue
                normalized_weight = normalized_total * (weight / total_aux_weight)
                blended_long += normalized_weight * float(np.clip(aux_long, 0.01, 0.99))
                blended_short += normalized_weight * float(np.clip(aux_short, 0.01, 0.99))
            long_probability = blended_long
            short_probability = blended_short
        if chronos_direction is not None and self.settings.model.chronos_probability_weight > 0:
            scale = max(self.settings.model.chronos_deadband * 4, 1e-6)
            bias = float(np.tanh(chronos_direction / scale) * 0.25)
            chronos_long = np.clip(0.5 + bias, 0.01, 0.99)
            chronos_short = np.clip(0.5 - bias, 0.01, 0.99)
            chronos_weight = float(np.clip(self.settings.model.chronos_probability_weight, 0.0, 0.5))
            long_probability = ((1.0 - chronos_weight) * long_probability) + (chronos_weight * chronos_long)
            short_probability = ((1.0 - chronos_weight) * short_probability) + (chronos_weight * chronos_short)
        return float(np.clip(long_probability, 0.01, 0.99)), float(np.clip(short_probability, 0.01, 0.99))

    def _write_optimization_profile(self, symbol: str, tuning: dict[str, Any]) -> None:
        payload = {
            "long_threshold": float(tuning["long_threshold"]),
            "short_threshold": float(tuning["short_threshold"]),
            "risk_per_trade": float(tuning["risk_per_trade"]),
            "atr_stop_loss_mult": float(tuning["atr_stop_loss_mult"]),
            "atr_take_profit_mult": float(tuning["atr_take_profit_mult"]),
            "lightgbm_probability_weight": float(tuning.get("lightgbm_probability_weight", 0.0)),
            "sequence_probability_weight": float(tuning.get("sequence_probability_weight", 0.0)),
            "transformer_probability_weight": float(tuning.get("transformer_probability_weight", 0.0)),
            "objective": float(tuning["objective"]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.settings.optimization_profile_path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _write_runtime_profile(self, symbol: str, payload: dict[str, Any]) -> str:
        path = self.settings.optimization_profile_path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup_path = path.with_name(f"{path.stem}.bak{path.suffix}")
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return str(path)

    def _chronos_gate_active(self, symbol: str, feature_frame) -> bool:
        if not self.settings.model.chronos_enabled:
            return False
        if feature_frame.empty:
            return False
        if not self.settings.model.chronos_selective_enabled:
            return True
        row = feature_frame.iloc[-1]
        atr_pct = float(row.get("atr_pct", 0.0) or 0.0)
        trend_strength = abs(float(row.get("ema_trend_strength", 0.0) or 0.0))
        spread_pct = float(row.get("spread_pct", 0.0) or 0.0)
        if atr_pct < float(self.settings.model.chronos_min_atr_pct):
            return False
        if atr_pct > float(self.settings.model.chronos_max_atr_pct):
            return False
        if trend_strength > float(self.settings.model.chronos_max_abs_ema_trend_strength):
            return False
        if spread_pct > float(self.settings.model.chronos_max_spread_pct):
            return False
        allowed_sessions = {str(item).lower() for item in self.settings.model.chronos_allowed_sessions}
        if not allowed_sessions or "all" in allowed_sessions:
            return True
        session_flags = {
            "tokyo": bool(row.get("session_tokyo", 0)),
            "london": bool(row.get("session_london", 0)),
            "new_york": bool(row.get("session_new_york", 0)),
            "overlap": bool(row.get("session_overlap", 0)),
        }
        return any(session_flags.get(session, False) for session in allowed_sessions)

    def _chronos_direction(self, symbol: str, feature_frame):
        if not self.settings.model.chronos_enabled:
            return None
        if not self._chronos_gate_active(symbol, feature_frame):
            return None
        try:
            return self.chronos.forecast_direction(
                symbol=symbol,
                frame=feature_frame[["time", "close"]],
                timeframe=self.settings.trading.timeframe,
                prediction_length=self.settings.model.chronos_prediction_length,
            )
        except Exception as exc:
            logger.warning("Chronos forecast failed for {}: {}", symbol, exc)
            return None

    def _log_cycle_skip(
        self,
        symbol: str,
        broker_symbol: str,
        reason: str,
        spread_points: int | None = None,
        spread_limits: dict[str, Any] | None = None,
    ) -> None:
        if spread_points is None or spread_limits is None:
            logger.info("Cycle {} ({}) skipped: {}", symbol, broker_symbol, reason)
            return
        logger.info(
            "Cycle {} ({}) skipped: {} spread={}/{} regime={}",
            symbol,
            broker_symbol,
            reason,
            spread_points,
            spread_limits.get("soft_limit_points"),
            spread_limits.get("runtime_regime_name"),
        )

    def backfill(self, symbol: str | None = None) -> dict[str, Any]:
        self.connect()
        summary: dict[str, Any] = {}
        count = max(self.settings.trading.history_bars, self.settings.trading.train_bars)
        for active_symbol in self._symbols(symbol):
            self.bridge.prepare_symbol(active_symbol)
            frame = self.bridge.fetch_rates(active_symbol, self.settings.trading.timeframe, count)
            upserted = self.database.upsert_bars(active_symbol, self.settings.trading.timeframe, frame)
            summary[active_symbol] = {"bars": int(len(frame)), "upserted": upserted}
        return summary

    def collect_latest(self, symbol: str | None = None) -> dict[str, Any]:
        self.connect()
        summary: dict[str, Any] = {}
        for active_symbol in self._symbols(symbol):
            self.bridge.prepare_symbol(active_symbol)
            frame = self.bridge.fetch_rates(
                active_symbol,
                self.settings.trading.timeframe,
                self.settings.trading.history_bars,
            )
            upserted = self.database.upsert_bars(active_symbol, self.settings.trading.timeframe, frame)
            summary[active_symbol] = {"bars": int(len(frame)), "upserted": upserted}
        return summary

    def heartbeat(self) -> dict[str, Any]:
        state = self.bridge.heartbeat()
        self.database.record_heartbeat(state.ok, state.latency_ms, state.message)
        if state.account:
            self.risk_manager.record_account_snapshot(state.account)

        if self._last_heartbeat_ok is not None and self._last_heartbeat_ok != state.ok:
            status_text = "restored" if state.ok else "lost"
            if state.ok:
                self.notifier.send_info(f"MT5 connectivity {status_text}: {state.message}")
            else:
                self.notifier.send_error(f"MT5 connectivity {status_text}: {state.message}")
        self._last_heartbeat_ok = state.ok
        return asdict(state)

    def refresh_models(self) -> dict[str, Any]:
        self.connect()
        data_summary = self.backfill()
        model_summary: dict[str, Any] = {}
        spread_summary: dict[str, Any] = {}
        for symbol in self._symbols():
            try:
                frame = self.database.load_bars(
                    symbol,
                    self.settings.trading.timeframe,
                    limit=max(self.settings.trading.train_bars, self.settings.trading.history_bars),
                )
                spread_summary[symbol] = self.spread_profile_manager.refresh_symbol(
                    symbol,
                    self.settings.trading.timeframe,
                    frame,
                )
                model_summary[symbol] = self.train_xgb(symbol=symbol)
            except Exception as exc:
                logger.exception("Model refresh failed for {}", symbol)
                self.database.record_event("ERROR", "model_refresh_failed", str(exc), {"symbol": symbol})
                self.notifier.send_error(f"Model refresh failed for {symbol}: {exc}")
                model_summary[symbol] = {"status": "failed", "reason": str(exc)}
                spread_summary[symbol] = spread_summary.get(symbol) or {"status": "failed", "reason": str(exc)}
        summary = {"data": data_summary, "spread_profiles": spread_summary, "models": model_summary}
        self.notifier.send_info(f"MT5AI model refresh completed for {','.join(self._symbols())}")
        return summary

    def inspect_broker_symbols(self, symbols: list[str] | None = None) -> dict[str, Any]:
        self.connect()
        requested_symbols = symbols or self._symbols()
        resolved = self.bridge.inspect_symbols(requested_symbols)
        return {
            "requested_symbols": requested_symbols,
            "resolved_symbols": resolved,
        }

    def train_xgb(self, symbol: str | None = None, side: str = "both") -> dict[str, Any]:
        target_symbol = self._symbols(symbol)[0]
        frame = self.database.load_bars(
            target_symbol,
            self.settings.trading.timeframe,
            limit=self.settings.trading.train_bars,
        )
        if frame.empty:
            raise ValueError(f"No stored data for {target_symbol}. Run backfill first.")

        supervised_frame = build_supervised_frame(
            frame,
            horizon=self.settings.model.positive_return_horizon,
            edge_bps=self.settings.model.positive_return_bps,
            breakout_pct=self.settings.model.volatility_breakout_pct,
            symbol=target_symbol,
        )
        if len(supervised_frame) < self.settings.model.training_min_rows:
            raise ValueError(
                f"Only {len(supervised_frame)} training rows available, require at least "
                f"{self.settings.model.training_min_rows}"
            )

        long_model = self._xgb_model(target_symbol, "long")
        short_model = self._xgb_model(target_symbol, "short")
        lgbm_long_model = (
            self._lgbm_model(target_symbol, "long") if self.settings.model.lightgbm_enabled else None
        )
        lgbm_short_model = (
            self._lgbm_model(target_symbol, "short") if self.settings.model.lightgbm_enabled else None
        )
        sequence_long_model = (
            self._sequence_model(target_symbol, "long") if self.settings.model.sequence_enabled else None
        )
        sequence_short_model = (
            self._sequence_model(target_symbol, "short") if self.settings.model.sequence_enabled else None
        )
        transformer_long_model = (
            self._transformer_model(target_symbol, "long") if self.settings.model.transformer_enabled else None
        )
        transformer_short_model = (
            self._transformer_model(target_symbol, "short") if self.settings.model.transformer_enabled else None
        )
        metrics = train_dual_models(
            symbol=target_symbol,
            supervised_frame=supervised_frame,
            settings=self.settings,
            backtest_engine=self.backtest_engine,
            long_model=long_model,
            short_model=short_model,
            lgbm_long_model=lgbm_long_model,
            lgbm_short_model=lgbm_short_model,
            sequence_long_model=sequence_long_model,
            sequence_short_model=sequence_short_model,
            transformer_long_model=transformer_long_model,
            transformer_short_model=transformer_short_model,
        )
        self._write_optimization_profile(target_symbol, metrics["threshold_tuning"])
        self.database.record_model_run(
            model_type="xgboost_long",
            symbol=target_symbol,
            timeframe=self.settings.trading.timeframe,
            artifact_path=str(self.settings.xgb_artifact_path_for(target_symbol, "long")),
            metrics=metrics["long"],
        )
        self.database.record_model_run(
            model_type="xgboost_short",
            symbol=target_symbol,
            timeframe=self.settings.trading.timeframe,
            artifact_path=str(self.settings.xgb_artifact_path_for(target_symbol, "short")),
            metrics=metrics["short"],
        )
        if self.settings.model.lightgbm_enabled:
            self.database.record_model_run(
                model_type="lightgbm_long",
                symbol=target_symbol,
                timeframe=self.settings.trading.timeframe,
                artifact_path=str(self.settings.lgbm_artifact_path_for(target_symbol, "long")),
                metrics=metrics["lightgbm_long"],
            )
            self.database.record_model_run(
                model_type="lightgbm_short",
                symbol=target_symbol,
                timeframe=self.settings.trading.timeframe,
                artifact_path=str(self.settings.lgbm_artifact_path_for(target_symbol, "short")),
                metrics=metrics["lightgbm_short"],
            )
        if self.settings.model.sequence_enabled:
            self.database.record_model_run(
                model_type="torch_lstm_long",
                symbol=target_symbol,
                timeframe=self.settings.trading.timeframe,
                artifact_path=str(self.settings.sequence_artifact_path_for(target_symbol, "long")),
                metrics=metrics["sequence_long"],
            )
            self.database.record_model_run(
                model_type="torch_lstm_short",
                symbol=target_symbol,
                timeframe=self.settings.trading.timeframe,
                artifact_path=str(self.settings.sequence_artifact_path_for(target_symbol, "short")),
                metrics=metrics["sequence_short"],
            )
        if self.settings.model.transformer_enabled:
            self.database.record_model_run(
                model_type="torch_transformer_long",
                symbol=target_symbol,
                timeframe=self.settings.trading.timeframe,
                artifact_path=str(self.settings.transformer_artifact_path_for(target_symbol, "long")),
                metrics=metrics["transformer_long"],
            )
            self.database.record_model_run(
                model_type="torch_transformer_short",
                symbol=target_symbol,
                timeframe=self.settings.trading.timeframe,
                artifact_path=str(self.settings.transformer_artifact_path_for(target_symbol, "short")),
                metrics=metrics["transformer_short"],
            )
        selected_side = side.lower()
        if selected_side in {"long", "short"}:
            return {
                "symbol": target_symbol,
                "side": selected_side,
                selected_side: metrics[selected_side],
                "threshold_tuning": metrics.get("threshold_tuning"),
                "splits": metrics.get("splits"),
            }
        return metrics

    def backtest(self, symbol: str | None = None) -> dict[str, Any]:
        target_symbol = self._symbols(symbol)[0]
        frame = self.database.load_bars(
            target_symbol,
            self.settings.trading.timeframe,
            limit=self.settings.trading.train_bars,
        )
        if frame.empty:
            raise ValueError(f"No stored data for {target_symbol}. Run backfill first.")
        long_model = self._xgb_model(target_symbol, "long")
        short_model = self._xgb_model(target_symbol, "short")
        lgbm_long_model = self._lgbm_model(target_symbol, "long") if self.settings.model.lightgbm_enabled else None
        lgbm_short_model = self._lgbm_model(target_symbol, "short") if self.settings.model.lightgbm_enabled else None
        long_model = long_model if long_model.is_ready() else None
        short_model = short_model if short_model.is_ready() else None
        lgbm_long_model = lgbm_long_model if lgbm_long_model is not None and lgbm_long_model.is_ready() else None
        lgbm_short_model = lgbm_short_model if lgbm_short_model is not None and lgbm_short_model.is_ready() else None
        stats = self.backtest_engine.run(
            target_symbol,
            frame,
            long_model=long_model,
            short_model=short_model,
            lgbm_long_model=lgbm_long_model,
            lgbm_short_model=lgbm_short_model,
        )
        stats["symbol"] = target_symbol
        return stats

    def walk_forward(self, symbol: str | None = None) -> dict[str, Any]:
        target_symbol = self._symbols(symbol)[0]
        frame = self.database.load_bars(
            target_symbol,
            self.settings.trading.timeframe,
            limit=self.settings.trading.train_bars,
        )
        if frame.empty:
            raise ValueError(f"No stored data for {target_symbol}. Run backfill first.")
        report = self.walk_forward_engine.run(target_symbol, frame)
        report["symbol"] = target_symbol
        return report

    def walk_forward_many(
        self,
        symbols: list[str] | None = None,
        write_report: bool = False,
        apply_profile: bool = False,
    ) -> dict[str, Any]:
        selected_symbols = symbols or self._symbols()
        results: dict[str, Any] = {}
        for target_symbol in selected_symbols:
            frame = self.database.load_bars(
                target_symbol,
                self.settings.trading.timeframe,
                limit=self.settings.trading.train_bars,
            )
            if frame.empty:
                raise ValueError(f"No stored data for {target_symbol}. Run backfill first.")
            report = self.walk_forward_engine.run(target_symbol, frame)
            report["symbol"] = target_symbol
            outcome: dict[str, Any] = {"report": report}
            if write_report:
                outcome["report_path"] = str(self.walk_forward_engine.write_report(target_symbol, report))
            if apply_profile:
                profile = self.walk_forward_engine.recommend_profile(target_symbol, report)
                outcome["profile"] = profile
                outcome["profile_path"] = self._write_runtime_profile(target_symbol, profile)
            results[target_symbol] = outcome
        return results

    def run_cycle(self) -> list[dict[str, Any]]:
        self.connect()
        self._watchdog_or_raise()
        account = self.bridge.account_info()
        if account is None:
            raise RuntimeError("Unable to retrieve account info")

        kill_switch_halted, kill_switch = self.risk_manager.evaluate_kill_switch(account)
        if kill_switch_halted and kill_switch is not None:
            reason = str(kill_switch["reason"])
            logger.error(reason)
            if kill_switch.get("newly_tripped", False):
                self.database.record_event(
                    "ERROR",
                    str(kill_switch["event_code"]),
                    reason,
                    kill_switch,
                )
                if self.settings.execution.close_all_on_guard_trip:
                    self.execution_engine.close_all_positions(reason)
                self.notifier.send_critical(reason)
            return [{"status": "halted", "reason": reason, "kill_switch": kill_switch}]

        reference_frames = self._reference_frames()
        results: list[dict[str, Any]] = []
        for symbol in self._symbols():
            try:
                symbol_info = self.bridge.prepare_symbol(symbol)
                broker_symbol = str(symbol_info.get("broker_symbol", symbol))
                market_frame = self.bridge.fetch_rates(
                    symbol,
                    self.settings.trading.timeframe,
                    self.settings.trading.history_bars,
                )
                self.database.upsert_bars(symbol, self.settings.trading.timeframe, market_frame)
                spread_points = self.risk_manager.spread_points(symbol_info)
                current_time = datetime.now(timezone.utc)
                news_guard = self.news_filter.evaluate(symbol, now=current_time)
                spread_limits = self.spread_profile_manager.evaluate(
                    symbol,
                    self.settings.trading.timeframe,
                    current_time=current_time,
                    frame=market_frame,
                    news_phase=news_guard.phase,
                )

                if self.risk_manager.spread_is_extreme(symbol_info, limit_points=spread_limits.extreme_limit_points):
                    reason = (
                        f"Extreme spread kill-switch for {symbol}: "
                        f"{spread_points} >= {spread_limits.extreme_limit_points}"
                    )
                    if self.settings.risk.close_positions_on_extreme_spread:
                        self.execution_engine.close_symbol_positions(symbol, reason)
                    self.database.record_event(
                        "WARNING",
                        "extreme_spread_guard",
                        reason,
                        {"symbol": symbol, "spread": spread_points, **spread_limits.payload()},
                    )
                    results.append(
                        {
                            "symbol": symbol,
                            "broker_symbol": broker_symbol,
                            "status": "skipped",
                            "reason": reason,
                            "spread_points": spread_points,
                            "spread_limits": spread_limits.payload(),
                        }
                    )
                    self._log_cycle_skip(
                        symbol,
                        broker_symbol,
                        reason,
                        spread_points=spread_points,
                        spread_limits=spread_limits.payload(),
                    )
                    continue

                effective_news_limit = (
                    min(news_guard.strict_spread_limit_points, spread_limits.news_limit_points)
                    if news_guard.strict_spread_limit_points is not None
                    else spread_limits.news_limit_points
                )
                if news_guard.blocked or news_guard.close_positions:
                    self.database.record_event(
                        "INFO",
                        "news_guard",
                        news_guard.reason,
                        news_guard.payload(),
                    )

                if (
                    news_guard.strict_spread_limit_points is not None
                    and not self.risk_manager.spread_is_safe(symbol_info, limit_points=effective_news_limit)
                ):
                    reason = (
                        f"{news_guard.reason};spread={spread_points};"
                        f"limit={effective_news_limit}"
                    )
                    if news_guard.blocked:
                        self.execution_engine.close_symbol_positions(symbol, reason)
                    self.database.record_event(
                        "WARNING",
                        "news_spread_guard",
                        reason,
                        {
                            "symbol": symbol,
                            "spread": spread_points,
                            "effective_news_limit": effective_news_limit,
                            **news_guard.payload(),
                            **spread_limits.payload(),
                        },
                    )
                    results.append(
                        {
                            "symbol": symbol,
                            "broker_symbol": broker_symbol,
                            "status": "skipped",
                            "reason": reason,
                            "spread_points": spread_points,
                            "spread_limits": spread_limits.payload(),
                        }
                    )
                    self._log_cycle_skip(
                        symbol,
                        broker_symbol,
                        reason,
                        spread_points=spread_points,
                        spread_limits=spread_limits.payload(),
                    )
                    continue

                if news_guard.long_blocked:
                    self.execution_engine.close_symbol_long_positions(symbol, news_guard.reason)

                if news_guard.close_positions:
                    self.execution_engine.close_symbol_positions(symbol, news_guard.reason)

                if news_guard.blocked:
                    results.append(
                        {
                            "symbol": symbol,
                            "broker_symbol": broker_symbol,
                            "status": "skipped",
                            "reason": news_guard.reason,
                            "spread_points": spread_points,
                            "spread_limits": spread_limits.payload(),
                        }
                    )
                    self._log_cycle_skip(
                        symbol,
                        broker_symbol,
                        news_guard.reason,
                        spread_points=spread_points,
                        spread_limits=spread_limits.payload(),
                    )
                    continue

                if not self.risk_manager.spread_is_safe(symbol_info, limit_points=spread_limits.soft_limit_points):
                    reason = (
                        f"Dynamic spread guard for {symbol}: "
                        f"{spread_points} > {spread_limits.soft_limit_points} "
                        f"({spread_limits.session_name}/{spread_limits.source})"
                    )
                    self.database.record_event(
                        "WARNING",
                        "dynamic_spread_guard",
                        reason,
                        {"symbol": symbol, "spread": spread_points, **spread_limits.payload()},
                    )
                    results.append(
                        {
                            "symbol": symbol,
                            "broker_symbol": broker_symbol,
                            "status": "skipped",
                            "reason": reason,
                            "spread_points": spread_points,
                            "spread_limits": spread_limits.payload(),
                        }
                    )
                    self._log_cycle_skip(
                        symbol,
                        broker_symbol,
                        reason,
                        spread_points=spread_points,
                        spread_limits=spread_limits.payload(),
                    )
                    continue

                feature_frame = build_feature_frame(
                    market_frame,
                    symbol=symbol,
                    reference_frames=reference_frames,
                )
                usable = feature_frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
                if usable.empty:
                    self._log_cycle_skip(symbol, broker_symbol, "insufficient_features")
                    results.append({"symbol": symbol, "status": "skipped", "reason": "insufficient_features"})
                    continue

                xgb_long_probability, xgb_short_probability = self._xgb_probabilities(symbol, usable)
                lgbm_long_probability, lgbm_short_probability = self._lgbm_probabilities(symbol, usable)
                sequence_long_probability, sequence_short_probability = self._sequence_probabilities(symbol, usable)
                transformer_long_probability, transformer_short_probability = self._transformer_probabilities(symbol, usable)
                chronos_direction = self._chronos_direction(symbol, usable)
                long_probability, short_probability = self._blend_live_probabilities(
                    symbol,
                    xgb_long=xgb_long_probability,
                    xgb_short=xgb_short_probability,
                    lgbm_long=lgbm_long_probability,
                    lgbm_short=lgbm_short_probability,
                    sequence_long=sequence_long_probability,
                    sequence_short=sequence_short_probability,
                    transformer_long=transformer_long_probability,
                    transformer_short=transformer_short_probability,
                    chronos_direction=chronos_direction,
                )
                long_threshold, short_threshold = self._xgb_thresholds(symbol)
                signal = self.signal_engine.evaluate(
                    symbol=symbol,
                    timeframe=self.settings.trading.timeframe,
                    feature_frame=usable,
                    long_probability=long_probability,
                    short_probability=short_probability,
                    chronos_direction=chronos_direction,
                    long_threshold=long_threshold,
                    short_threshold=short_threshold,
                )

                management_actions = self.execution_engine.manage_open_positions(symbol, market_frame)
                
                outcome = self.execution_engine.execute(signal, usable)
                result = {
                    "symbol": symbol,
                    "broker_symbol": broker_symbol,
                    "signal": signal.side,
                    "probability": signal.probability,
                    "long_probability": signal.long_probability,
                    "short_probability": signal.short_probability,
                    "xgb_long_probability": xgb_long_probability,
                    "xgb_short_probability": xgb_short_probability,
                    "lightgbm_long_probability": lgbm_long_probability,
                    "lightgbm_short_probability": lgbm_short_probability,
                    "sequence_long_probability": sequence_long_probability,
                    "sequence_short_probability": sequence_short_probability,
                    "transformer_long_probability": transformer_long_probability,
                    "transformer_short_probability": transformer_short_probability,
                    "long_threshold": long_threshold,
                    "short_threshold": short_threshold,
                    "chronos_direction": signal.chronos_direction,
                    "reason": signal.reason,
                    "news_guard": news_guard.reason,
                    "execution_status": outcome.status,
                    "execution_message": outcome.message,
                    "spread_points": spread_points,
                    "spread_limits": spread_limits.payload(),
                    "spread_regime": spread_limits.runtime_regime_name,
                    "position_management": management_actions,
                }
                results.append(result)
                logger.info(
                    "Cycle {} ({}) -> signal={} probability={:.3f} outcome={} news={} spread={}/{} regime={}",
                    symbol,
                    broker_symbol,
                    signal.side,
                    signal.probability,
                    outcome.status,
                    news_guard.reason,
                    spread_points,
                    spread_limits.soft_limit_points,
                    spread_limits.runtime_regime_name,
                )
            except Exception as exc:
                logger.exception("Cycle failed for {}", symbol)
                self.database.record_event("ERROR", "cycle_failure", str(exc), {"symbol": symbol})
                self.notifier.send_error(f"MT5AI cycle failed for {symbol}: {exc}")
                results.append({"symbol": symbol, "status": "failed", "reason": str(exc)})
        return results

    def run_scheduler(self) -> None:
        if not self._run_lock.acquire():
            raise RuntimeError(f"Another run-live process already holds {self.settings.run_lock_path}")
        try:
            self.connect()
            heartbeat_state = self.heartbeat()
            logger.info("Initial heartbeat: {}", heartbeat_state)

            scheduler = BlockingScheduler(timezone="UTC")
            trade_trigger = CronTrigger(**scheduler_trigger_args(self.settings.trading.timeframe))
            scheduler.add_job(
                self.run_cycle,
                trigger=trade_trigger,
                id="trade_cycle",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
                misfire_grace_time=30,
            )
            scheduler.add_job(
                self.heartbeat,
                trigger="interval",
                seconds=self.settings.bot.heartbeat_seconds,
                id="heartbeat",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
                misfire_grace_time=15,
            )
            scheduler.add_job(
                self.refresh_models,
                trigger=CronTrigger(
                    hour=self.settings.bot.model_refresh_hour_utc,
                    minute=self.settings.bot.model_refresh_minute_utc,
                    second=15,
                ),
                id="model_refresh",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
                misfire_grace_time=180,
            )
            logger.info(
                "Scheduler started for timeframe {} with symbols {}",
                self.settings.trading.timeframe,
                ",".join(self.settings.trading.symbols),
            )
            self.notifier.send_info(
                f"MT5AI run-live started for {','.join(self.settings.trading.symbols)} on "
                f"{self.settings.trading.timeframe}"
            )
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")
        except Exception as exc:
            self.notifier.send_critical(f"MT5AI run-live stopped unexpectedly: {exc}")
            raise
        finally:
            self._run_lock.release()

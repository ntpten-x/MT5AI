from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from config import Settings
from modules.db import Database
from modules.mt5_bridge import MT5Bridge
from modules.notifications import TelegramNotifier
from modules.risk_control import RiskManager
from modules.types import ExecutionOutcome, TradeSignal


class ExecutionEngine:
    def __init__(
        self,
        settings: Settings,
        bridge: MT5Bridge,
        database: Database,
        risk_manager: RiskManager,
        notifier: TelegramNotifier,
    ):
        self.settings = settings
        self.bridge = bridge
        self.database = database
        self.risk_manager = risk_manager
        self.notifier = notifier

    def _success_retcodes(self, mt5) -> set[int]:
        return {
            int(mt5.TRADE_RETCODE_DONE),
            int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", mt5.TRADE_RETCODE_DONE)),
            int(getattr(mt5, "TRADE_RETCODE_PLACED", mt5.TRADE_RETCODE_DONE)),
        }

    def _retryable_retcodes(self, mt5) -> set[int]:
        codes = {
            int(getattr(mt5, "TRADE_RETCODE_REQUOTE", -1)),
            int(getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", -2)),
            int(getattr(mt5, "TRADE_RETCODE_OFF_QUOTES", -3)),
            int(getattr(mt5, "TRADE_RETCODE_CONNECTION", -4)),
        }
        return {code for code in codes if code >= 0}

    def _build_order_request(
        self,
        signal: TradeSignal,
        symbol_info: dict[str, Any],
        tick: dict[str, Any],
        volume: float,
        sl: float,
        tp: float,
        filling_mode: int | None = None,
    ) -> dict[str, Any]:
        mt5 = self.bridge.module
        digits = int(symbol_info.get("digits", 5))
        price_step = float(symbol_info.get("trade_tick_size") or symbol_info.get("point") or 0.0)
        raw_price = tick["ask"] if signal.side == "long" else tick["bid"]
        price = self.risk_manager.round_step(raw_price, price_step, precision=digits)
        order_type = mt5.ORDER_TYPE_BUY if signal.side == "long" else mt5.ORDER_TYPE_SELL
        selected_filling_mode = (
            int(filling_mode)
            if filling_mode is not None
            else int(self.bridge.filling_modes(symbol_info=symbol_info)[0])
        )
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.settings.execution.deviation_points,
            "magic": 26_012_026,
            "comment": f"{self.settings.execution.comment_prefix}:{signal.side}:{signal.timeframe}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": selected_filling_mode,
        }

    def _position_size_below_minimum_message(
        self,
        signal: TradeSignal,
        symbol_info: dict[str, Any],
        account: dict[str, Any],
        entry_price: float,
        stop_price: float,
        risk_fraction: float | None = None,
    ) -> str:
        applied_risk_fraction = (
            float(risk_fraction)
            if risk_fraction is not None
            else float(signal.risk_fraction if signal.risk_fraction is not None else self.settings.risk_per_trade_for(signal.symbol))
        )
        risk_amount = float(account.get("equity", 0.0)) * applied_risk_fraction
        price_distance = abs(entry_price - stop_price)
        tick_size = float(symbol_info.get("trade_tick_size") or symbol_info.get("point") or 0.0)
        tick_value = float(symbol_info.get("trade_tick_value") or 0.0)
        money_per_lot = 0.0
        if tick_size > 0 and tick_value > 0:
            money_per_lot = (price_distance / tick_size) * tick_value
        if money_per_lot <= 0:
            contract_size = float(symbol_info.get("trade_contract_size") or 100_000.0)
            money_per_lot = price_distance * contract_size
        minimum_volume = float(symbol_info.get("volume_min") or symbol_info.get("volume_step") or 0.0)
        minimum_risk_amount = money_per_lot * minimum_volume if money_per_lot > 0 and minimum_volume > 0 else 0.0
        return (
            f"Calculated volume below broker minimum for {signal.symbol} "
            f"(risk_fraction={applied_risk_fraction:.4f}, risk_budget={risk_amount:.4f}, volume_min={minimum_volume:.2f}, "
            f"min_risk_amount={minimum_risk_amount:.4f})"
        )

    def execute(self, signal: TradeSignal, feature_frame: pd.DataFrame) -> ExecutionOutcome:
        if signal.side == "flat":
            return ExecutionOutcome(status="skipped", message="No tradable signal")

        positions = self.bridge.positions_get(symbol=signal.symbol)
        if len(positions) >= self.settings.trading.max_positions_per_symbol:
            return ExecutionOutcome(status="skipped", message="Max positions per symbol reached")

        last_trade_time = self.database.get_last_trade_opened_at(signal.symbol)
        if last_trade_time is not None:
            elapsed = (datetime.now(timezone.utc) - last_trade_time).total_seconds()
            if elapsed < self.settings.trading.min_trade_interval_seconds:
                return ExecutionOutcome(
                    status="skipped",
                    message=(
                        "Min trade interval active "
                        f"({elapsed:.0f}s < {self.settings.trading.min_trade_interval_seconds}s)"
                    ),
                )

        symbol_info = self.bridge.symbol_info(signal.symbol)
        if not self.risk_manager.spread_is_safe(symbol_info, symbol=signal.symbol):
            message = f"Spread too wide for {signal.symbol}: {symbol_info.get('spread', 0)}"
            self.database.record_event("WARNING", "spread_filter", message, {"symbol": signal.symbol})
            return ExecutionOutcome(status="skipped", message=message)

        account = self.bridge.account_info()
        if account is None:
            return ExecutionOutcome(status="failed", message="Account info unavailable")

        tick = self.bridge.symbol_tick(signal.symbol)
        atr = float(feature_frame["atr_14"].iloc[-1])
        entry_price = tick["ask"] if signal.side == "long" else tick["bid"]
        sl, tp = self.risk_manager.build_levels(signal.symbol, signal.side, entry_price, atr, symbol_info)
        applied_risk_fraction = (
            float(signal.risk_fraction)
            if signal.risk_fraction is not None
            else self.settings.risk_per_trade_for(signal.symbol)
        )
        volume = self.risk_manager.calculate_volume(
            signal.symbol,
            symbol_info,
            account,
            entry_price,
            sl,
            risk_fraction=applied_risk_fraction,
        )
        if volume <= 0:
            message = self._position_size_below_minimum_message(
                signal,
                symbol_info,
                account,
                entry_price,
                sl,
                risk_fraction=applied_risk_fraction,
            )
            self.database.record_event(
                "WARNING",
                "position_size_below_minimum",
                message,
                {
                    "symbol": signal.symbol,
                    "entry_price": entry_price,
                    "stop_price": sl,
                    "equity": float(account.get("equity", 0.0)),
                    "risk_per_trade": applied_risk_fraction,
                },
            )
            return ExecutionOutcome(status="skipped", message=message)

        request = self._build_order_request(signal, symbol_info, tick, volume, sl, tp)
        synthetic_ticket = int(time.time() * 1000)

        if self.settings.execution.dry_run:
            self.database.record_trade(
                ticket=synthetic_ticket,
                provider="MT5",
                symbol=signal.symbol,
                side=signal.side,
                volume=volume,
                requested_price=request["price"],
                executed_price=request["price"],
                sl=sl,
                tp=tp,
                status="dry_run",
                strategy=signal.reason,
                comment=request["comment"],
                request_payload=request,
                response_payload={"mode": "dry_run"},
            )
            return ExecutionOutcome(
                status="dry_run",
                message="Order simulated only",
                ticket=synthetic_ticket,
                requested_volume=volume,
                executed_price=request["price"],
                payload=request,
            )

        mt5 = self.bridge.module
        invalid_fill_retcode = int(getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030))
        filling_modes = self.bridge.filling_modes(symbol_info=symbol_info)
        for attempt in range(1, self.settings.execution.max_retries + 1):
            try:
                tick = self.bridge.symbol_tick(signal.symbol)
                for index, filling_mode in enumerate(filling_modes):
                    request = self._build_order_request(signal, symbol_info, tick, volume, sl, tp, filling_mode=filling_mode)
                    result = self.bridge.order_send(request)
                    retcode = int(result.get("retcode", -1))
                    ticket = int(result.get("order") or result.get("deal") or synthetic_ticket)

                    if retcode in self._success_retcodes(mt5):
                        executed_price = float(result.get("price") or request["price"])
                        self.database.record_trade(
                            ticket=ticket,
                            provider="MT5",
                            symbol=signal.symbol,
                            side=signal.side,
                            volume=volume,
                            requested_price=request["price"],
                            executed_price=executed_price,
                            sl=sl,
                            tp=tp,
                            status="filled",
                            strategy=signal.reason,
                            comment=request["comment"],
                            request_payload=request,
                            response_payload=result,
                        )
                        return ExecutionOutcome(
                            status="filled",
                            message="Order filled",
                            ticket=ticket,
                            retcode=retcode,
                            requested_volume=volume,
                            executed_price=executed_price,
                            payload=result,
                        )

                    if retcode == invalid_fill_retcode and index + 1 < len(filling_modes):
                        logger.warning(
                            "MT5 invalid fill mode {} for {}; retrying with alternate mode",
                            filling_mode,
                            signal.symbol,
                        )
                        continue

                    if retcode not in self._retryable_retcodes(mt5):
                        self.database.record_event(
                            "ERROR",
                            "order_rejected",
                            f"MT5 rejected {signal.symbol} with retcode {retcode}",
                            {"request": request, "response": result},
                        )
                        self.notifier.send_error(
                            f"MT5 rejected {signal.symbol} with retcode {retcode}"
                        )
                        return ExecutionOutcome(
                            status="rejected",
                            message=f"MT5 rejected order with retcode {retcode}",
                            ticket=ticket,
                            retcode=retcode,
                            requested_volume=volume,
                            payload=result,
                        )

                    logger.warning(
                        "Retryable MT5 retcode {} for {} on attempt {}",
                        retcode,
                        signal.symbol,
                        attempt,
                    )
                    time.sleep(self.settings.execution.retry_delay_seconds)
                    break
            except Exception as exc:
                logger.warning("Execution attempt {} failed for {}: {}", attempt, signal.symbol, exc)
                if attempt >= self.settings.execution.max_retries:
                    break
                time.sleep(self.settings.execution.retry_delay_seconds)

        message = f"Execution failed after {self.settings.execution.max_retries} attempts"
        self.database.record_event("ERROR", "execution_failed", message, {"symbol": signal.symbol})
        self.notifier.send_error(message)
        return ExecutionOutcome(status="failed", message=message, requested_volume=volume, payload=request)

    def close_symbol_positions(self, symbol: str, reason: str) -> list[dict[str, Any]]:
        return self._close_positions(self.bridge.positions_get(symbol=symbol), reason)

    def close_all_positions(self, reason: str) -> list[dict[str, Any]]:
        return self._close_positions(self.bridge.positions_get(), reason)

    def _close_positions(self, positions: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for position in positions:
            try:
                result = self.bridge.close_position(
                    position=position,
                    deviation=self.settings.execution.deviation_points,
                    comment=f"{self.settings.execution.comment_prefix}:guard",
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Failed to close position {}: {}", position.get("ticket"), exc)
        if results:
            self.database.record_event(
                "WARNING",
                "positions_closed_by_guard",
                reason,
                {"count": len(results), "tickets": [item.get("order") for item in results]},
            )
            self.notifier.send_warning(f"Guardian closed {len(results)} positions. Reason: {reason}")
        return results

    def close_symbol_long_positions(self, symbol: str, reason: str) -> list[dict[str, Any]]:
        mt5 = self.bridge.module
        long_positions = [
            position
            for position in self.bridge.positions_get(symbol=symbol)
            if int(position.get("type", -1)) == int(mt5.POSITION_TYPE_BUY)
        ]
        return self._close_positions(long_positions, reason)

    def manage_open_positions(self, symbol: str, market_frame: pd.DataFrame) -> list[dict[str, Any]]:
        trailing_trigger_rr = self.settings.trailing_trigger_rr_for(symbol)
        trailing_lock_rr = self.settings.trailing_lock_rr_for(symbol)
        time_exit_bars = self.settings.time_exit_bars_for(symbol)
        time_exit_min_progress_rr = self.settings.time_exit_min_progress_rr_for(symbol)
        if (
            not self.settings.risk.auto_breakeven_enable
            and time_exit_bars <= 0
            and not (trailing_trigger_rr > 0 and trailing_lock_rr > 0)
        ):
            return []

        positions = self.bridge.positions_get(symbol=symbol)
        if not positions:
            return []

        symbol_info = self.bridge.symbol_info(symbol)
        results: list[dict[str, Any]] = []
        for position in positions:
            try:
                ticket = position.get("ticket")
                position_type = position.get("type")
                entry_price = float(position.get("price_open", 0))
                current_price = float(position.get("price_current", 0))
                sl = float(position.get("sl", 0))
                tp = float(position.get("tp", 0))
                open_time = position.get("time")

                if entry_price <= 0 or current_price <= 0:
                    continue

                is_long = position_type == 0
                profit_distance = (current_price - entry_price) if is_long else (entry_price - current_price)
                risk_distance = abs(entry_price - sl) if sl > 0 else 0
                profit_rr = (profit_distance / risk_distance) if risk_distance > 0 else 0.0
                breakeven_rr = self.settings.auto_breakeven_rr_for(symbol)
                digits = int(position.get("digits") or symbol_info.get("digits", 5))
                price_step = float(
                    position.get("trade_tick_size")
                    or symbol_info.get("trade_tick_size")
                    or symbol_info.get("point")
                    or 0.0
                )

                if trailing_trigger_rr > 0 and trailing_lock_rr > 0 and risk_distance > 0 and profit_rr >= trailing_trigger_rr:
                    if is_long:
                        target_sl = entry_price + (risk_distance * trailing_lock_rr)
                        should_modify = target_sl > sl
                    else:
                        target_sl = entry_price - (risk_distance * trailing_lock_rr)
                        should_modify = sl <= 0 or target_sl < sl
                    if should_modify:
                        new_sl = self.risk_manager.round_step(target_sl, price_step, precision=digits)
                        result = self.bridge.modify_position(
                            ticket=ticket,
                            sl=new_sl,
                            tp=tp,
                            comment=f"{self.settings.execution.comment_prefix}:trail",
                        )
                        if result:
                            results.append(
                                {
                                    "ticket": ticket,
                                    "action": "trail",
                                    "new_sl": new_sl,
                                    "profit_rr": round(profit_rr, 4),
                                }
                            )
                            logger.info(
                                "Trailed position {} to {} at {:.2f}R",
                                ticket,
                                new_sl,
                                profit_rr,
                            )
                            sl = new_sl
                            risk_distance = abs(entry_price - sl)

                if self.settings.risk.auto_breakeven_enable and risk_distance > 0:
                    if profit_rr >= breakeven_rr:
                        if (is_long and sl < entry_price) or (not is_long and sl > entry_price):
                            new_sl = entry_price
                            new_sl = self.risk_manager.round_step(new_sl, price_step, precision=digits)
                            result = self.bridge.modify_position(
                                ticket=ticket,
                                sl=new_sl,
                                tp=tp,
                                comment=f"{self.settings.execution.comment_prefix}:breakeven",
                            )
                            if result:
                                results.append({"ticket": ticket, "action": "breakeven", "new_sl": new_sl})
                                logger.info("Moved position {} to breakeven at {}", ticket, new_sl)

                if time_exit_bars > 0 and open_time:
                    bars_since_open = self._count_bars_since_open(open_time, market_frame)
                    if bars_since_open >= time_exit_bars:
                        if abs(profit_distance) < (abs(entry_price - sl) * time_exit_min_progress_rr):
                            result = self.bridge.close_position(
                                position=position,
                                deviation=self.settings.execution.deviation_points,
                                comment=f"{self.settings.execution.comment_prefix}:time_exit",
                            )
                            if result:
                                results.append({"ticket": ticket, "action": "time_exit", "bars": bars_since_open})
                                logger.info("Time exit for position {} after {} bars", ticket, bars_since_open)

            except Exception as exc:
                logger.warning("Failed to manage position {}: {}", position.get("ticket"), exc)

        return results

    def _count_bars_since_open(self, open_time: int, market_frame: pd.DataFrame) -> int:
        if market_frame.empty:
            return 0
        try:
            open_dt = datetime.fromtimestamp(open_time, tz=timezone.utc)
            bars_after = market_frame[market_frame["time"] >= open_dt]
            return len(bars_after)
        except Exception:
            return 0

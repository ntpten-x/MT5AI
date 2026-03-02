from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from config import Settings
from modules.db import Database


class RiskManager:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database

    def record_account_snapshot(self, account: dict[str, Any]) -> None:
        self.database.record_equity_snapshot(
            account,
            datetime.now(timezone.utc),
            symbol="ACCOUNT",
            provider="MT5",
        )

    def _is_crypto_symbol(self, symbol: str) -> bool:
        return symbol.upper() in {item.upper() for item in self.settings.trading.crypto_symbols}

    def round_step(self, value: float, step: float, precision: int | None = None) -> float:
        if step <= 0:
            if precision is None:
                return value
            return round(value, precision)
        steps = round(value / step)
        rounded = steps * step
        if precision is None:
            text = f"{step:.10f}".rstrip("0")
            precision = len(text.split(".")[1]) if "." in text else 0
        return round(rounded, precision)

    def daily_drawdown_exceeded(self, account: dict[str, Any]) -> tuple[bool, float]:
        now = datetime.now(timezone.utc)
        self.database.record_equity_snapshot(account, now, symbol="ACCOUNT", provider="MT5")
        baseline = self.database.get_daily_equity_baseline(now.date())
        if baseline is None:
            return False, 0.0

        baseline_equity = float(baseline["equity"])
        if baseline_equity <= 0:
            return False, 0.0

        current_equity = float(account.get("equity", baseline_equity))
        drawdown = max(0.0, (baseline_equity - current_equity) / baseline_equity)
        return drawdown >= self.settings.risk.daily_drawdown_limit, drawdown

    def equity_high_watermark_exceeded(self, account: dict[str, Any]) -> tuple[bool, float]:
        now = datetime.now(timezone.utc)
        high_watermark = self.database.get_daily_high_watermark_equity(now.date())
        if high_watermark is None:
            return False, 0.0

        if high_watermark <= 0:
            return False, 0.0

        current_equity = float(account.get("equity", high_watermark))
        drawdown_from_high = max(0.0, (high_watermark - current_equity) / high_watermark)
        return drawdown_from_high >= self.settings.risk.daily_equity_max_drawdown_pct, drawdown_from_high

    def consecutive_losses_exceeded(self) -> tuple[bool, int]:
        consecutive_losses = self.database.count_consecutive_losses(limit=20)
        exceeded = consecutive_losses >= self.settings.risk.max_consecutive_losses
        return exceeded, consecutive_losses

    def spread_points(self, symbol_info: dict[str, Any]) -> int:
        return int(symbol_info.get("spread", 0))

    def spread_limit_points(self, symbol: str | None = None) -> int:
        if symbol and self._is_crypto_symbol(symbol):
            return int(self.settings.risk.crypto_spread_limit_points)
        return int(self.settings.risk.spread_limit_points)

    def news_spread_limit_points(self, symbol: str | None = None) -> int:
        if symbol and self._is_crypto_symbol(symbol):
            return int(self.settings.risk.crypto_news_spread_limit_points)
        return int(self.settings.risk.news_spread_limit_points)

    def extreme_spread_limit_points(self, symbol: str | None = None) -> int:
        if symbol and self._is_crypto_symbol(symbol):
            return int(self.settings.risk.crypto_extreme_spread_limit_points)
        return int(self.settings.risk.extreme_spread_limit_points)

    def spread_is_safe(
        self,
        symbol_info: dict[str, Any],
        symbol: str | None = None,
        limit_points: int | None = None,
    ) -> bool:
        spread = self.spread_points(symbol_info)
        max_points = limit_points if limit_points is not None else self.spread_limit_points(symbol)
        return spread <= max_points

    def spread_is_extreme(
        self,
        symbol_info: dict[str, Any],
        symbol: str | None = None,
        limit_points: int | None = None,
    ) -> bool:
        spread = int(symbol_info.get("spread", 0))
        max_points = limit_points if limit_points is not None else self.extreme_spread_limit_points(symbol)
        return spread >= max_points

    def _round_price(self, symbol_info: dict[str, Any], value: float) -> float:
        digits = int(symbol_info.get("digits", 5))
        price_step = float(symbol_info.get("trade_tick_size") or symbol_info.get("point") or 0.0)
        return self.round_step(value, price_step, precision=digits)

    def build_levels(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        atr: float,
        symbol_info: dict[str, Any],
    ) -> tuple[float, float]:
        point = float(symbol_info.get("point", 0.00001))
        atr_sl_mult = self.settings.atr_stop_loss_mult_for(symbol)
        atr_tp_mult = self.settings.atr_take_profit_mult_for(symbol)
        atr_distance = max(point * 10, atr * atr_sl_mult)
        tp_distance = max(point * 10, atr * atr_tp_mult)

        min_distance = max(0.0, float(symbol_info.get("trade_stops_level", 0)) * point)
        atr_distance = max(atr_distance, min_distance)
        tp_distance = max(tp_distance, min_distance)

        if side == "long":
            sl = entry_price - atr_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + atr_distance
            tp = entry_price - tp_distance

        return self._round_price(symbol_info, sl), self._round_price(symbol_info, tp)

    def auto_breakeven_rr(self, symbol: str) -> float:
        if self._is_crypto_symbol(symbol):
            return float(self.settings.risk.auto_breakeven_rr_crypto)
        return float(self.settings.risk.auto_breakeven_rr_default)

    def calculate_volume(
        self,
        symbol: str,
        symbol_info: dict[str, Any],
        account: dict[str, Any],
        entry_price: float,
        stop_price: float,
    ) -> float:
        risk_amount = float(account.get("equity", 0.0)) * self.settings.risk_per_trade_for(symbol)
        price_distance = abs(entry_price - stop_price)
        if risk_amount <= 0 or price_distance <= 0:
            return 0.0

        tick_size = float(symbol_info.get("trade_tick_size") or symbol_info.get("point") or 0.0)
        tick_value = float(symbol_info.get("trade_tick_value") or 0.0)
        money_per_lot = 0.0
        if tick_size > 0 and tick_value > 0:
            ticks = price_distance / tick_size
            money_per_lot = ticks * tick_value

        if money_per_lot <= 0:
            contract_size = float(symbol_info.get("trade_contract_size") or 100_000.0)
            money_per_lot = price_distance * contract_size
        if money_per_lot <= 0:
            return 0.0

        raw_volume = risk_amount / money_per_lot
        step = float(symbol_info.get("volume_step") or 0.01)
        minimum = float(symbol_info.get("volume_min") or step)
        maximum = float(symbol_info.get("volume_max") or raw_volume)

        steps = math.floor(raw_volume / step)
        normalized = steps * step
        clipped = min(max(normalized, minimum), maximum)
        step_text = f"{step:.10f}".rstrip("0")
        precision = len(step_text.split(".")[1]) if "." in step_text else 0
        return self.round_step(clipped, step, precision=precision)

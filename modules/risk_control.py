from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from config import Settings
from modules.db import Database


@dataclass
class KillSwitchState:
    active: bool
    event_code: str
    reason: str
    metric_name: str
    metric_value: float
    threshold: float
    tripped_at: str


class RiskManager:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self._kill_switch: KillSwitchState | None = None
        self._load_kill_switch()

    def _persist_kill_switch(self) -> None:
        path = self.settings.kill_switch_path
        if self._kill_switch is None:
            if path.exists():
                path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self._kill_switch), ensure_ascii=True, indent=2), encoding="utf-8")

    def _load_kill_switch(self) -> None:
        path = self.settings.kill_switch_path
        if not path.exists():
            self._kill_switch = None
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._kill_switch = None
            return
        if not isinstance(payload, dict):
            self._kill_switch = None
            return
        try:
            self._kill_switch = KillSwitchState(
                active=bool(payload.get("active", True)),
                event_code=str(payload.get("event_code", "kill_switch_unknown")),
                reason=str(payload.get("reason", "Kill switch is active")),
                metric_name=str(payload.get("metric_name", "unknown")),
                metric_value=float(payload.get("metric_value", 0.0)),
                threshold=float(payload.get("threshold", 0.0)),
                tripped_at=str(payload.get("tripped_at", "")),
            )
        except Exception:
            self._kill_switch = None

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

    def kill_switch_active(self) -> bool:
        self._load_kill_switch()
        return self._kill_switch is not None and self._kill_switch.active

    def kill_switch_status(self) -> dict[str, Any] | None:
        self._load_kill_switch()
        if self._kill_switch is None:
            return None
        return asdict(self._kill_switch)

    def reset_kill_switch(self) -> None:
        self._kill_switch = None
        self._persist_kill_switch()

    def _trip_kill_switch(
        self,
        *,
        event_code: str,
        reason: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> dict[str, Any]:
        if self._kill_switch is None:
            self._kill_switch = KillSwitchState(
                active=True,
                event_code=event_code,
                reason=reason,
                metric_name=metric_name,
                metric_value=float(metric_value),
                threshold=float(threshold),
                tripped_at=datetime.now(timezone.utc).isoformat(),
            )
            self._persist_kill_switch()
        payload = asdict(self._kill_switch)
        payload["newly_tripped"] = True
        return payload

    def evaluate_kill_switch(self, account: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        self._load_kill_switch()
        if self._kill_switch is not None:
            payload = asdict(self._kill_switch)
            payload["newly_tripped"] = False
            return True, payload

        drawdown_halted, drawdown = self.daily_drawdown_exceeded(account)
        if drawdown_halted:
            threshold = float(self.settings.risk.daily_drawdown_limit)
            reason = f"Daily drawdown kill-switch tripped at {drawdown:.2%} (limit {threshold:.2%})"
            return True, self._trip_kill_switch(
                event_code="kill_switch_daily_drawdown",
                reason=reason,
                metric_name="daily_drawdown",
                metric_value=drawdown,
                threshold=threshold,
            )

        equity_halted, equity_drawdown = self.equity_high_watermark_exceeded(account)
        if equity_halted:
            threshold = float(self.settings.risk.daily_equity_max_drawdown_pct)
            reason = (
                f"Equity high-watermark kill-switch tripped at {equity_drawdown:.2%} "
                f"(limit {threshold:.2%})"
            )
            return True, self._trip_kill_switch(
                event_code="kill_switch_equity_high_watermark",
                reason=reason,
                metric_name="equity_high_watermark_drawdown",
                metric_value=equity_drawdown,
                threshold=threshold,
            )

        loss_halted, consecutive_losses = self.consecutive_losses_exceeded()
        if loss_halted:
            threshold = float(self.settings.risk.max_consecutive_losses)
            reason = (
                f"Consecutive-loss kill-switch tripped at {consecutive_losses:.0f} losses "
                f"(limit {threshold:.0f})"
            )
            return True, self._trip_kill_switch(
                event_code="kill_switch_consecutive_losses",
                reason=reason,
                metric_name="consecutive_losses",
                metric_value=float(consecutive_losses),
                threshold=threshold,
            )

        return False, None

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
        risk_fraction: float | None = None,
    ) -> float:
        fraction = float(risk_fraction) if risk_fraction is not None else self.settings.risk_per_trade_for(symbol)
        risk_amount = float(account.get("equity", 0.0)) * fraction
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

        if raw_volume + 1e-12 < minimum:
            return 0.0

        steps = math.floor(raw_volume / step)
        normalized = steps * step
        clipped = min(max(normalized, minimum), maximum)
        step_text = f"{step:.10f}".rstrip("0")
        precision = len(step_text.split(".")[1]) if "." in step_text else 0
        return self.round_step(clipped, step, precision=precision)

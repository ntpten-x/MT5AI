from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
from uuid import uuid4


@dataclass(slots=True, frozen=True)
class TradingSafetyPolicy:
    enabled: bool = True
    kill_switch_enabled: bool = False
    manual_approval_required: bool = False
    max_order_notional_usd: float = 5_000.0
    max_order_qty: float = 1_000.0
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    allow_market_without_quote: bool = False


@dataclass(slots=True, frozen=True)
class TradingSafetyDecision:
    allowed: bool
    approval_required: bool
    reason: str
    order_id: str | None = None
    estimated_notional: float | None = None
    message: str = ""


class TradingSafetyStore:
    """File-backed execution safety policy, pending approvals, and audit log."""

    def __init__(
        self,
        *,
        path: Path,
        policy: TradingSafetyPolicy | None = None,
    ) -> None:
        self.path = path
        self.policy = policy or TradingSafetyPolicy()
        self._lock = RLock()
        self._state: dict[str, Any] = {
            "kill_switch_enabled": self.policy.kill_switch_enabled,
            "pending_orders": {},
            "audit": [],
            "day_key": None,
            "day_start_equity": None,
            "peak_equity": None,
        }
        self._load()
        if self.policy.kill_switch_enabled:
            self._state["kill_switch_enabled"] = True
            self._persist()

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending = dict(self._state.get("pending_orders") or {})
            return {
                "available": True,
                "enabled": self.policy.enabled,
                "kill_switch_enabled": bool(self._state.get("kill_switch_enabled")),
                "manual_approval_required": self.policy.manual_approval_required,
                "max_order_notional_usd": self.policy.max_order_notional_usd,
                "max_order_qty": self.policy.max_order_qty,
                "daily_loss_limit_pct": self.policy.daily_loss_limit_pct,
                "max_drawdown_pct": self.policy.max_drawdown_pct,
                "allow_market_without_quote": self.policy.allow_market_without_quote,
                "pending_order_count": sum(1 for item in pending.values() if isinstance(item, Mapping) and item.get("status") == "pending"),
                "day_start_equity": self._state.get("day_start_equity"),
                "peak_equity": self._state.get("peak_equity"),
                "last_audit": (self._state.get("audit") or [])[-1] if self._state.get("audit") else None,
            }

    def set_kill_switch(self, enabled: bool, *, reason: str = "", actor: str = "") -> None:
        with self._lock:
            self._state["kill_switch_enabled"] = bool(enabled)
            self._append_audit_locked(
                {
                    "event": "kill_switch_changed",
                    "enabled": bool(enabled),
                    "reason": reason,
                    "actor": actor,
                }
            )
            self._persist()

    def evaluate_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        limit_price: float | None,
        estimated_price: float | None,
        account_equity: float | None,
        cash: float | None,
        source: str,
        actor: str,
        approved: bool = False,
        existing_order_id: str | None = None,
    ) -> TradingSafetyDecision:
        now = datetime.now(timezone.utc)
        normalized_symbol = str(symbol or "").strip().upper()
        normalized_side = str(side or "").strip().lower()
        normalized_type = str(order_type or "market").strip().lower()
        price = self._coerce_float(limit_price) if limit_price is not None else self._coerce_float(estimated_price)
        estimated_notional = (float(qty) * price) if price is not None and price > 0 else None
        with self._lock:
            self._update_equity_locked(account_equity=account_equity, now=now)
            decision = self._evaluate_locked(
                symbol=normalized_symbol,
                side=normalized_side,
                qty=float(qty),
                order_type=normalized_type,
                price=price,
                estimated_notional=estimated_notional,
                account_equity=account_equity,
                cash=cash,
                approved=approved,
            )
            if decision is not None:
                self._append_audit_locked(
                    {
                        "event": "order_blocked",
                        "symbol": normalized_symbol,
                        "side": normalized_side,
                        "qty": float(qty),
                        "order_type": normalized_type,
                        "estimated_notional": estimated_notional,
                        "reason": decision.reason,
                        "source": source,
                        "actor": actor,
                    }
                )
                self._persist()
                return decision
            if self.policy.manual_approval_required and not approved:
                order_id = existing_order_id or f"safe-{uuid4().hex[:12]}"
                pending_orders = dict(self._state.get("pending_orders") or {})
                pending_orders[order_id] = {
                    "order_id": order_id,
                    "symbol": normalized_symbol,
                    "side": normalized_side,
                    "qty": float(qty),
                    "order_type": normalized_type,
                    "limit_price": limit_price,
                    "estimated_price": price,
                    "estimated_notional": estimated_notional,
                    "source": source,
                    "actor": actor,
                    "status": "pending",
                    "created_at": now.isoformat(),
                }
                self._state["pending_orders"] = pending_orders
                self._append_audit_locked({"event": "order_pending_approval", "order_id": order_id, "symbol": normalized_symbol, "side": normalized_side, "source": source, "actor": actor})
                self._persist()
                return TradingSafetyDecision(
                    allowed=False,
                    approval_required=True,
                    reason="manual_approval_required",
                    order_id=order_id,
                    estimated_notional=estimated_notional,
                    message=f"Order requires manual approval: {order_id}",
                )
            self._append_audit_locked(
                {
                    "event": "order_approved_by_policy",
                    "order_id": existing_order_id,
                    "symbol": normalized_symbol,
                    "side": normalized_side,
                    "qty": float(qty),
                    "order_type": normalized_type,
                    "estimated_notional": estimated_notional,
                    "source": source,
                    "actor": actor,
                    "approved": approved,
                }
            )
            self._persist()
            return TradingSafetyDecision(
                allowed=True,
                approval_required=False,
                reason="allowed",
                order_id=existing_order_id,
                estimated_notional=estimated_notional,
                message="Order allowed by safety policy",
            )

    def get_pending_order(self, order_id: str) -> dict[str, Any] | None:
        with self._lock:
            pending = self._state.get("pending_orders")
            if not isinstance(pending, dict):
                return None
            item = pending.get(str(order_id).strip())
            return dict(item) if isinstance(item, Mapping) else None

    def list_pending_orders(self, *, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            pending = self._state.get("pending_orders")
            if not isinstance(pending, dict):
                return []
            rows = [dict(item) for item in pending.values() if isinstance(item, Mapping) and item.get("status") == "pending"]
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return rows[: max(1, int(limit))]

    def mark_order_status(
        self,
        order_id: str,
        *,
        status: str,
        broker_order_id: str | None = None,
        reason: str = "",
        actor: str = "",
    ) -> None:
        with self._lock:
            pending = dict(self._state.get("pending_orders") or {})
            item = dict(pending.get(order_id) or {})
            if item:
                item["status"] = str(status or "").strip().lower() or "unknown"
                item["broker_order_id"] = broker_order_id
                item["updated_at"] = datetime.now(timezone.utc).isoformat()
                item["status_reason"] = reason
                pending[order_id] = item
                self._state["pending_orders"] = pending
            self._append_audit_locked(
                {
                    "event": "pending_order_status_changed",
                    "order_id": order_id,
                    "status": status,
                    "broker_order_id": broker_order_id,
                    "reason": reason,
                    "actor": actor,
                }
            )
            self._persist()

    def record_execution(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        broker_order_id: str | None,
        status: str | None,
        estimated_notional: float | None,
        source: str,
        actor: str,
    ) -> None:
        with self._lock:
            self._append_audit_locked(
                {
                    "event": "order_submitted",
                    "symbol": str(symbol or "").strip().upper(),
                    "side": str(side or "").strip().lower(),
                    "qty": float(qty),
                    "broker_order_id": broker_order_id,
                    "status": status,
                    "estimated_notional": estimated_notional,
                    "source": source,
                    "actor": actor,
                }
            )
            self._persist()

    def _evaluate_locked(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: float | None,
        estimated_notional: float | None,
        account_equity: float | None,
        cash: float | None,
        approved: bool,
    ) -> TradingSafetyDecision | None:
        if not self.policy.enabled:
            return None
        if bool(self._state.get("kill_switch_enabled")):
            return TradingSafetyDecision(False, False, "kill_switch_enabled", estimated_notional=estimated_notional, message="Trading kill switch is enabled")
        if not symbol or side not in {"buy", "sell"}:
            return TradingSafetyDecision(False, False, "invalid_order", estimated_notional=estimated_notional, message="Invalid symbol or side")
        if qty <= 0:
            return TradingSafetyDecision(False, False, "invalid_quantity", estimated_notional=estimated_notional, message="Quantity must be positive")
        if qty > self.policy.max_order_qty:
            return TradingSafetyDecision(False, False, "max_order_qty_exceeded", estimated_notional=estimated_notional, message="Quantity exceeds safety limit")
        if order_type == "market" and price is None and not self.policy.allow_market_without_quote:
            return TradingSafetyDecision(False, False, "missing_market_price", estimated_notional=estimated_notional, message="Market order needs a current quote for safety sizing")
        if estimated_notional is not None and estimated_notional > self.policy.max_order_notional_usd:
            return TradingSafetyDecision(False, False, "max_order_notional_exceeded", estimated_notional=estimated_notional, message="Order notional exceeds safety limit")
        if side == "buy" and estimated_notional is not None and cash is not None and estimated_notional > max(0.0, float(cash)) * 0.98:
            return TradingSafetyDecision(False, False, "insufficient_cash_safety", estimated_notional=estimated_notional, message="Estimated buy exceeds available cash")
        day_start = self._coerce_float(self._state.get("day_start_equity"))
        peak = self._coerce_float(self._state.get("peak_equity"))
        equity = self._coerce_float(account_equity)
        if equity is not None and day_start is not None and day_start > 0:
            if equity <= day_start * (1.0 - self.policy.daily_loss_limit_pct):
                return TradingSafetyDecision(False, False, "daily_loss_limit_triggered", estimated_notional=estimated_notional, message="Daily loss limit triggered")
        if equity is not None and peak is not None and peak > 0:
            if equity <= peak * (1.0 - self.policy.max_drawdown_pct):
                return TradingSafetyDecision(False, False, "max_drawdown_limit_triggered", estimated_notional=estimated_notional, message="Max drawdown limit triggered")
        return None

    def _update_equity_locked(self, *, account_equity: float | None, now: datetime) -> None:
        equity = self._coerce_float(account_equity)
        if equity is None or equity <= 0:
            return
        day_key = now.strftime("%Y-%m-%d")
        if self._state.get("day_key") != day_key:
            self._state["day_key"] = day_key
            self._state["day_start_equity"] = equity
        peak = self._coerce_float(self._state.get("peak_equity"))
        self._state["peak_equity"] = max(equity, peak or equity)
        self._state["last_equity"] = equity
        self._state["last_equity_at"] = now.isoformat()

    def _append_audit_locked(self, event: Mapping[str, Any]) -> None:
        audit = list(self._state.get("audit") or [])
        payload = dict(event)
        payload["at"] = datetime.now(timezone.utc).isoformat()
        audit.append(payload)
        self._state["audit"] = audit[-500:]

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if isinstance(payload, dict):
            self._state.update(payload)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
